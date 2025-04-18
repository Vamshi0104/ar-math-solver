// Updates the AR marker text with the extracted math expression.
window.updateARText = function (fullText) {
  const textEntity = document.querySelector('#mathText');
  const cleaned = (fullText || '').replace(/\n|\r/g, ' ').substring(0, 300);  // use 'cleaned'

  if (!textEntity) {
    console.warn('#mathText entity not found in the AR scene.');
    return;
  }

    textEntity.setAttribute('text', {
      value: cleaned,
      color: 'yellow',
      align: 'center',
      wrapCount:30,    
      width: 3.5,         // smaller AR box
      baseline: 'center',
      lineHeight: 50,
      font: 'https://cdn.aframe.io/fonts/Exo2Bold.fnt'
    });


  console.log('AR text updated:', cleaned);
};



// AR.js marker detection logging component
AFRAME.registerComponent("markerhandler", {
  init: function () {
    const marker = this.el;
    const textEntity = document.querySelector("#mathText");

    marker.addEventListener("markerFound", () => {
      console.log("Marker detected!");
      if (textEntity) {
        textEntity.setAttribute("visible", true);
        console.log("Math text is now visible.");
      }
    });

    marker.addEventListener("markerLost", () => {
      console.log("Marker lost.");
      if (textEntity) {
        textEntity.setAttribute("visible", false);
        console.log("Math text hidden.");
      }
    });
  }
});

