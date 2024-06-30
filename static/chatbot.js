const socket = io();

function sendMessage() {
    const message = document.getElementById('message').value;
    socket.send(message);
    document.getElementById('messages').innerHTML += `<div class="message user-message">${message}</div>`;
    document.getElementById('message').value = '';
}

socket.on('response', function(response) {
    document.getElementById('messages').innerHTML += `<div class="message bot-response">${response}</div>`;
});
