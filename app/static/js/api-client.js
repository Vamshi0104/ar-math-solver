document.querySelector("form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.querySelector("#imageInput");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const queryRes = await fetch("/api/equation/ar-query", {
      method: "POST",
      body: formData
    });

    const queryData = await queryRes.json();

    const raw = queryData.raw || 'N/A';
    const corrected = queryData.corrected || 'N/A';
    const finalQuery = queryData.query || 'N/A';
    const answer = queryData.answer ? `\n\nFinal Answer: ${queryData.answer}` : 'No answer provided';

    const cleanCorrected = corrected.replace(/\\\\/g, '\\').replace(/\\\(|\\\)/g, '').trim();
    const cleanAnswer = queryData.answer?.trim() || 'No answer provided';

    const arMessage = `${cleanCorrected}\n\n\n=> Final Answer:\n${cleanAnswer}`;

    console.log('Sending AR Text:', arMessage.replace(/\n/g, '\\n'));
    window.updateARText(arMessage);
  } catch (err) {
    console.error("Upload or processing failed", err);
    document.querySelector("#equationOutput").textContent = "Error extracting equation.";
  }
});
