
function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function app() {
  // Application code here
  if (outputMessageEl) {
    outputMessageEl.innerText = "TensorFlow.js version " + tf.version.tfjs;
  }
}

(async function initApp() {

  try {
    initTFJS();
    await app();

  } catch (error) {
    console.error(error);
    if (outputMessageEl) {
      outputMessageEl.innerText = error.message;
    }
  }

}());




