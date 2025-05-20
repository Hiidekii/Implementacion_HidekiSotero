const btn = document.getElementById("iniciarCaptura");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctxOverlay = overlay.getContext("2d");

btn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: "always" },
            audio: false
        });

        video.srcObject = stream;

        setInterval(() => {
            enviarFrame(video);
        }, 1000); // Enviar cada 1 segundo

    } catch (err) {
        console.error("Error al capturar pantalla:", err);
        alert("No se pudo iniciar la captura.");
    }
});

async function enviarFrame(video) {
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64 = canvas.toDataURL("image/jpeg");

    try {
        const response = await fetch("http://localhost:5000/detect_emotion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64 })
        });

        if (!response.ok) {
            throw new Error("Error en la respuesta del backend");
        }

        const data = await response.json();
        mostrarEmociones(data.result);
        dibujarBoundingBoxes(data.result);

    } catch (error) {
        console.error("❌ Error al enviar frame:", error);
    }
}

function mostrarEmociones(lista) {
    const contenedor = document.getElementById("emociones");
    contenedor.innerHTML = ""; // Limpiar contenido anterior

    lista.forEach(item => {
        const p = document.createElement("p");
        p.textContent = `${item.rostro} - ${item.emocion} - ${item.hora}`;
        contenedor.appendChild(p);
    });
}

function dibujarBoundingBoxes(lista) {
    // Asegurarse de que el canvas tenga el mismo tamaño que el video
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);

    lista.forEach(item => {
        if (item.box) {
            const { x, y, w, h } = item.box;
            ctxOverlay.strokeStyle = "#00FF00";
            ctxOverlay.lineWidth = 2;
            ctxOverlay.strokeRect(x, y, w, h);

            // Dibujar etiqueta de emoción arriba del cuadro
            ctxOverlay.fillStyle = "rgba(0,0,0,0.6)";
            ctxOverlay.fillRect(x, y - 25, w, 20);
            ctxOverlay.fillStyle = "#ffffff";
            ctxOverlay.font = "14px sans-serif";
            ctxOverlay.fillText(`${item.emocion}`, x + 5, y - 10);
        }
    });
}
