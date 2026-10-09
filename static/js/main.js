document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.add('hidden'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active to current
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-target');
            const targetContent = document.getElementById(targetId);
            targetContent.classList.remove('hidden');
            targetContent.classList.add('active');
        });
    });

    // Embed Form Submission
    const embedForm = document.getElementById('embed-form');
    embedForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = embedForm.querySelector('button');
        const originalText = btn.innerText;
        btn.innerText = 'Processing...';
        btn.disabled = true;

        const formData = new FormData();
        formData.append('image', document.getElementById('embed-image').files[0]);
        formData.append('message', document.getElementById('embed-message').value);
        formData.append('method', document.getElementById('embed-method').value);

        try {
            const response = await fetch('/api/embed', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'stego_image.png';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                
                const resultBox = document.getElementById('embed-result');
                resultBox.classList.remove('hidden');
                resultBox.innerHTML = '<p style="color:var(--accent-cover)">✅ Successfully embedded and downloaded!</p>';
            } else {
                const res = await response.json();
                alert('Error: ' + res.error);
            }
        } catch (error) {
            alert('An error occurred.');
            console.error(error);
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    });

    // Extract Form Submission
    const extractForm = document.getElementById('extract-form');
    extractForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = extractForm.querySelector('button');
        const originalText = btn.innerText;
        btn.innerText = 'Extracting...';
        btn.disabled = true;

        const formData = new FormData();
        formData.append('image', document.getElementById('extract-image').files[0]);
        formData.append('method', document.getElementById('extract-method').value);

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                const resultBox = document.getElementById('extract-result');
                resultBox.classList.remove('hidden');
                resultBox.querySelector('.message-display').innerText = data.message;
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('An error occurred.');
            console.error(error);
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    });

    // Analyze Form Submission
    const analyzeForm = document.getElementById('analyze-form');
    analyzeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = analyzeForm.querySelector('button');
        const originalText = btn.innerText;
        btn.innerText = 'Analyzing...';
        btn.disabled = true;

        const formData = new FormData();
        formData.append('image', document.getElementById('analyze-image').files[0]);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                const resultBox = document.getElementById('analyze-result');
                resultBox.classList.remove('hidden');
                
                const confPercent = Math.round(data.confidence * 100);
                
                // Update UI Ring
                const circle = document.querySelector('.circle');
                const text = document.querySelector('.percentage');
                const title = document.getElementById('pred-class');
                const confSpan = document.getElementById('pred-conf');
                const lapSpan = document.getElementById('pred-lap');
                
                circle.setAttribute('stroke-dasharray', `${confPercent}, 100`);
                text.textContent = `${confPercent}%`;
                
                title.textContent = data.prediction;
                confSpan.textContent = `${confPercent}%`;
                lapSpan.textContent = data.laplacian_variance.toFixed(2);
                
                // Colors
                circle.classList.remove('stego', 'cover');
                title.classList.remove('stego-text', 'cover-text');
                
                if (data.prediction === 'Stego') {
                    circle.classList.add('stego');
                    title.classList.add('stego-text');
                } else {
                    circle.classList.add('cover');
                    title.classList.add('cover-text');
                }

            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('An error occurred.');
            console.error(error);
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    });
});
