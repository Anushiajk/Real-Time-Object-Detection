document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById('video-feed');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                video.srcObject = stream;
            })
            .catch((error) => {
                console.log("Something went wrong with the webcam stream: ", error);
            });
    }
});
