document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const micBtn = document.getElementById('mic-btn');
    const assistantPanel = document.querySelector('.assistant-panel');
    const transcriptionBox = document.getElementById('transcription-box');
    const transcribedText = document.getElementById('transcribed-text');
    const loadingContainer = document.getElementById('loading-container');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const productsGrid = document.getElementById('products-grid');
    const resultsCount = document.getElementById('results-count');
    const productCardTemplate = document.getElementById('product-card-template');

    // Status and toast
    const systemStatus = document.getElementById('system-status');
    const toastMessage = document.getElementById('toast-message');
    const errorToast = document.getElementById('error-toast');

    // Text Search
    const textSearchInput = document.getElementById('text-search-input');
    const textSearchBtn = document.getElementById('text-search-btn');

    let audioContext;
    let mediaStreamSource;
    let processor;
    let isRecording = false;

    // We'll collect raw audio chunks (Float32Arrays) here
    let audioData = [];

    // Check if the browser supports getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        console.log('getUserMedia supported.');
    } else {
        showError('Voice recording is not supported in this browser.');
        micBtn.disabled = true;
        systemStatus.innerHTML = `<span class="pulse-dot"></span><span>Uninitialized</span>`;
        systemStatus.classList.add('error');
    }

    // Convert Float32Array to 16-bit PCM WAV
    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');

        // FMT sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);          // Subchunk1Size (16 for PCM)
        view.setUint16(20, 1, true);           // AudioFormat (1 for PCM)
        view.setUint16(22, 1, true);           // NumChannels (1 for Mono)
        view.setUint32(24, sampleRate, true);  // SampleRate
        view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
        view.setUint16(32, 2, true);           // BlockAlign (NumChannels * BitsPerSample/8)
        view.setUint16(34, 16, true);          // BitsPerSample (16 for 16-bit)

        // Data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        // Write PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            mediaStreamSource = audioContext.createMediaStreamSource(stream);

            // ScriptProcessorNode is deprecated but highly compatible for manual audio extracting
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                // clone the array because it gets reused
                audioData.push(new Float32Array(inputData));
            };

            mediaStreamSource.connect(processor);
            processor.connect(audioContext.destination);

            // Store stream so we can stop it later
            audioContext.stream = stream;

        } catch (err) {
            console.error("Microphone access error:", err);
            showError('Microphone access denied or error occurred.');
            throw err;
        }
    }

    function stopRecordingAndProcess() {
        // Disconnect and clean up
        if (processor && mediaStreamSource) {
            processor.disconnect();
            mediaStreamSource.disconnect();
        }

        if (audioContext && audioContext.stream) {
            audioContext.stream.getTracks().forEach(track => track.stop());
            audioContext.close();
        }

        // Merge all Float32Arrays into one
        let totalLength = audioData.reduce((acc, arr) => acc + arr.length, 0);
        let mergedData = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < audioData.length; i++) {
            mergedData.set(audioData[i], offset);
            offset += audioData[i].length;
        }

        // Encode to WAV and send
        const wavBlob = encodeWAV(mergedData, 16000); // We forced 16kHz
        processAudioBlob(wavBlob);
    }

    // --- Actions ---

    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                // Clear old data and start
                audioData = [];
                isRecording = true;
                await startRecording();

                // UI Updates
                micBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
                assistantPanel.classList.add('listening');
                transcriptionBox.style.display = 'flex';
                transcribedText.textContent = "Listening to you...";

                // Hide old results
                recommendationsContainer.style.display = 'none';
            } catch (err) {
                isRecording = false;
            }
        } else {
            // Stop recording
            isRecording = false;
            stopRecordingAndProcess();

            // UI Updates
            micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            assistantPanel.classList.remove('listening');
            assistantPanel.classList.add('compact'); // Shrink the hero area
            transcribedText.innerHTML = "Processing audio... <i class='fa-solid fa-spinner fa-spin' style='margin-left: 10px; opacity: 0.5;'></i>";

            showLoading();
        }
    });

    // Handle Text Search Fallback
    textSearchBtn.addEventListener('click', () => {
        const query = textSearchInput.value.trim();
        if (query) {
            assistantPanel.classList.add('compact');
            transcriptionBox.style.display = 'flex';
            transcribedText.textContent = query;
            textSearchInput.value = '';

            showLoading();
            recommendationsContainer.style.display = 'none';

            fetch('/api/search_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            })
                .then(res => res.json())
                .then(handleApiResponse)
                .catch(err => {
                    hideLoading();
                    showError('Search failed: ' + err.message);
                    transcribedText.textContent = "Error occurred.";
                });
        }
    });

    textSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            textSearchBtn.click();
        }
    });

    function processAudioBlob(blob) {
        const formData = new FormData();
        // The backend expects an audio file. We append the blob.
        formData.append('audio', blob, 'recording.wav');

        fetch('/api/process_audio', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(handleApiResponse)
            .catch(error => {
                console.error('Error:', error);
                hideLoading();
                showError("Failed to communicate with the server.");
                transcribedText.textContent = "Processing failed.";
            });
    }

    function handleApiResponse(data) {
        hideLoading();

        if (!data.success) {
            showError(data.error || "Unknown error occurred");
            transcribedText.innerHTML = `${data.text ? data.text + " (Error)" : "Could not understand."}`;
            return;
        }

        // Update transcription text
        transcribedText.textContent = data.text;

        // Render matches
        renderProducts(data.recommendations);
    }

    function renderProducts(products) {
        productsGrid.innerHTML = '';

        if (!products || products.length === 0) {
            resultsCount.textContent = "No products found matches";
            resultsCount.style.background = "rgba(239, 68, 68, 0.1)";
            resultsCount.style.color = "var(--danger)";
            recommendationsContainer.style.display = 'flex';
            return;
        }

        resultsCount.textContent = `Found ${products.length} matches`;
        resultsCount.style.background = "rgba(255, 255, 255, 0.05)";
        resultsCount.style.color = "var(--text-muted)";

        products.forEach((prod, index) => {
            const clone = productCardTemplate.content.cloneNode(true);
            const card = clone.querySelector('.product-card');

            // Stagger animations
            card.style.animationDelay = `${index * 0.1}s`;

            // Populate data
            const imgEl = clone.querySelector('.product-img');
            imgEl.src = prod.image || 'https://via.placeholder.com/300x200?text=No+Image';

            // Score mapping (converting cosine similarity to percentage)
            const scorePct = Math.round(prod.similarity_score * 100);
            clone.querySelector('.score-val').textContent = scorePct;

            // Title and metadata
            clone.querySelector('.product-title').textContent = prod.name;
            clone.querySelector('.product-category').textContent = prod.category || prod.sub_category || 'General';

            // Ratings - handle 'N/A' or empty
            const ratingEl = clone.querySelector('.rating-val');
            ratingEl.textContent = prod.rating && prod.rating !== 'N/A' ? prod.rating : 'New';

            // Pricing
            clone.querySelector('.discount-price').textContent = prod.price || 'Check price';
            clone.querySelector('.actual-price').textContent = prod.actual_price !== prod.price ? (prod.actual_price || '') : '';

            // Link
            clone.querySelector('.buy-btn').href = prod.link || '#';

            productsGrid.appendChild(clone);
        });

        recommendationsContainer.style.display = 'flex';
    }

    // --- Helpers ---
    function showLoading() {
        loadingContainer.style.display = 'block';
    }

    function hideLoading() {
        loadingContainer.style.display = 'none';
    }

    function showError(msg) {
        toastMessage.textContent = msg;
        errorToast.classList.add('show');
        setTimeout(() => {
            errorToast.classList.remove('show');
        }, 4000);
    }
});
