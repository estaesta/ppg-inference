document.addEventListener("DOMContentLoaded", function () {
  const socket = io();
  let names = [];

  socket.on("monitor", function (data) {
    const monitor = document.getElementById("monitor");
    if (names.includes(data.name)) {
      const card = document.getElementById(data.name);
      card.querySelector(".stress").innerHTML = data.stress;
      card.querySelector(".hr").innerHTML = data.hr;
    } else {
      names.push(data.name);
      const template = `
      <div class="card p-3" id="${data.name}">
        <h2>${data.name}</h2>
        <div class="row">
          <div class="p-3">
            <h3>Stress Level:</h3>
            <div class="stress fs-5">${data.stress}</div>
          </div>
          <div class="row">
            <div class="p-3 d-flex gap-3">
              <h3>HR:</h3>
              <div class="hr fs-5">${data.hr}</div>
            </div>
          </div>
        </div>
      </div>
      `;
      monitor.innerHTML += template;
    }
    const card = document.getElementById(data.name);
    const stress = card.querySelector(".stress");
    if (data.stress === "stress") {
      stress.style.color = "red";
    } else if (data.stress === "amusement") {
      stress.style.color = "green";
    } else {
      stress.style.color = "black";
    }
    
  });
});
