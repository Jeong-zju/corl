document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;
  const dialog = document.getElementById("media-dialog");
  const dialogTitle = document.getElementById("media-dialog-title");
  const dialogVideo = document.getElementById("dialog-video");
  const dialogImage = document.getElementById("dialog-image");
  const loadingText = document.querySelector("[data-loading-text]");

  function cleanupDialog() {
    dialogVideo.pause();
    dialogVideo.removeAttribute("src");
    dialogVideo.load();
    dialogVideo.hidden = true;

    dialogImage.removeAttribute("src");
    dialogImage.alt = "";
    dialogImage.hidden = true;
  }

  function openDialog(type, src, title) {
    cleanupDialog();
    dialogTitle.textContent = title || "预览";

    if (type === "image") {
      dialogImage.src = src;
      dialogImage.alt = title || "预览";
      dialogImage.hidden = false;
    } else {
      dialogVideo.src = src;
      dialogVideo.hidden = false;
      dialogVideo.load();
    }

    dialog.showModal();
  }

  document.querySelectorAll("[data-open-media]").forEach((element) => {
    element.addEventListener("click", () => {
      openDialog(
        element.dataset.openMedia || "video",
        element.dataset.src || "",
        element.dataset.title || "预览"
      );
    });
  });

  document.querySelector("[data-close-dialog]")?.addEventListener("click", () => {
    dialog.close();
  });

  dialog?.addEventListener("click", (event) => {
    const rect = dialog.getBoundingClientRect();
    const clickedInside =
      rect.top <= event.clientY &&
      event.clientY <= rect.bottom &&
      rect.left <= event.clientX &&
      event.clientX <= rect.right;

    if (!clickedInside) {
      dialog.close();
    }
  });

  dialog?.addEventListener("close", cleanupDialog);

  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", (event) => {
      const submitter = event.submitter;
      if (loadingText && submitter?.dataset.loadingText) {
        loadingText.textContent = submitter.dataset.loadingText;
      } else if (loadingText) {
        loadingText.textContent = "加载中...";
      }
      body.classList.add("is-loading");
    });
  });
});
