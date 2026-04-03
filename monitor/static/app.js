document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;
  const dialog = document.getElementById("media-dialog");
  const dialogTitle = document.getElementById("media-dialog-title");
  const dialogVideo = document.getElementById("dialog-video");
  const dialogImage = document.getElementById("dialog-image");
  const loadingText = document.querySelector("[data-loading-text]");

  function cleanupDialog() {
    if (!dialogVideo || !dialogImage) {
      return;
    }

    dialogVideo.pause();
    dialogVideo.removeAttribute("src");
    dialogVideo.load();
    dialogVideo.hidden = true;

    dialogImage.removeAttribute("src");
    dialogImage.alt = "";
    dialogImage.hidden = true;
  }

  function openDialog(type, src, title) {
    if (!dialog || !dialogTitle) {
      return;
    }

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

  function buildVideoUrl(relativePath) {
    const encodedPath = String(relativePath || "")
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    return `/video/${encodedPath}`;
  }

  function createButton({ label, className, onClick, disabled = false, active = false }) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.disabled = disabled;
    button.setAttribute("aria-disabled", String(disabled));

    if (active) {
      button.classList.add("is-active");
      button.setAttribute("aria-current", "page");
    }

    if (typeof onClick === "function" && !disabled) {
      button.addEventListener("click", onClick);
    }

    return button;
  }

  function createVideoCard(runPayload, video) {
    const card = document.createElement("section");
    card.className = "video-card";

    const header = document.createElement("div");
    header.className = "video-card-header";

    const titleRow = document.createElement("div");
    titleRow.className = "video-title-row";

    const title = document.createElement("h3");
    title.textContent = video.episode_label || video.name || "未命名视频";
    title.title = video.name || video.episode_label || "视频";
    titleRow.appendChild(title);

    const statusBadge = document.createElement("span");
    statusBadge.className = `status-badge ${video.status_class || "status-unknown"}`;
    statusBadge.textContent = video.status_label || "未知";
    titleRow.appendChild(statusBadge);

    const meta = document.createElement("p");
    meta.className = "video-subtitle";
    meta.textContent = video.name || "";

    header.appendChild(titleRow);
    header.appendChild(meta);
    card.appendChild(header);

    const mediaUrl = buildVideoUrl(video.relative_path);
    const mediaTitle = `${runPayload.env} / ${runPayload.algorithm} / ${video.task_label} / ${video.episode_label}`;

    if (video.is_gif) {
      const image = document.createElement("img");
      image.src = mediaUrl;
      image.alt = mediaTitle;
      image.className = "media-preview";
      image.loading = "lazy";
      image.dataset.openMedia = "image";
      image.dataset.src = mediaUrl;
      image.dataset.title = mediaTitle;
      card.appendChild(image);
    } else {
      const preview = document.createElement("video");
      preview.className = "media-preview";
      preview.controls = true;
      preview.preload = "metadata";
      preview.playsInline = true;
      preview.dataset.openMedia = "video";
      preview.dataset.src = mediaUrl;
      preview.dataset.title = mediaTitle;

      const source = document.createElement("source");
      source.src = mediaUrl;
      preview.appendChild(source);
      card.appendChild(preview);
    }

    if (video.status_key === "failure" && video.failure_reason) {
      const reasonBox = document.createElement("div");
      reasonBox.className = "failure-reason-box";

      const reasonLabel = document.createElement("span");
      reasonLabel.className = "failure-reason-label";
      reasonLabel.textContent = "失败原因";

      const reasonText = document.createElement("strong");
      reasonText.className = "failure-reason-text";
      reasonText.textContent = video.failure_reason;

      reasonBox.appendChild(reasonLabel);
      reasonBox.appendChild(reasonText);
      card.appendChild(reasonBox);
    }

    const expandButton = document.createElement("button");
    expandButton.type = "button";
    expandButton.className = "ghost-button";
    expandButton.textContent = "放大查看";
    expandButton.dataset.openMedia = video.is_gif ? "image" : "video";
    expandButton.dataset.src = mediaUrl;
    expandButton.dataset.title = mediaTitle;
    card.appendChild(expandButton);

    return card;
  }

  function initializeRunColumn(runElement) {
    const payloadNode = runElement.querySelector("[data-run-payload]");
    const tabsNode = runElement.querySelector("[data-task-tabs]");
    const summaryNode = runElement.querySelector("[data-task-summary]");
    const listNode = runElement.querySelector("[data-video-list]");
    const paginationNode = runElement.querySelector("[data-pagination]");
    const filterButtons = Array.from(runElement.querySelectorAll("[data-status-filter]"));

    if (!payloadNode || !tabsNode || !summaryNode || !listNode || !paginationNode) {
      return;
    }

    const runPayload = JSON.parse(payloadNode.textContent || "{}");
    const taskGroups = Array.isArray(runPayload.taskGroups) ? runPayload.taskGroups : [];
    if (taskGroups.length === 0) {
      return;
    }

    const pageSize = Number.parseInt(runElement.dataset.videosPerPage || "6", 10) || 6;
    const state = {
      taskKey: taskGroups[0].task_key,
      status: "all",
      page: 1,
    };

    function getCurrentTask() {
      return (
        taskGroups.find((taskGroup) => taskGroup.task_key === state.taskKey) ||
        taskGroups[0]
      );
    }

    function getFilteredVideos(taskGroup) {
      const videos = Array.isArray(taskGroup.videos) ? taskGroup.videos : [];
      if (state.status === "all") {
        return videos;
      }
      return videos.filter((video) => video.status_key === state.status);
    }

    function renderTabs() {
      tabsNode.replaceChildren();

      taskGroups.forEach((taskGroup) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "task-tab";
        tab.dataset.taskKey = taskGroup.task_key;
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", String(taskGroup.task_key === state.taskKey));

        if (taskGroup.task_key === state.taskKey) {
          tab.classList.add("is-active");
        }

        const label = document.createElement("span");
        label.className = "task-tab-label";
        label.textContent = taskGroup.task_label;

        const counts = document.createElement("span");
        counts.className = "task-tab-counts";
        counts.textContent = `${taskGroup.success_count}/${taskGroup.video_count}`;

        tab.appendChild(label);
        tab.appendChild(counts);
        tab.addEventListener("click", () => {
          if (state.taskKey === taskGroup.task_key) {
            return;
          }
          state.taskKey = taskGroup.task_key;
          state.page = 1;
          render();
        });

        tabsNode.appendChild(tab);
      });
    }

    function renderStatusFilters() {
      filterButtons.forEach((button) => {
        const isActive = button.dataset.statusFilter === state.status;
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", String(isActive));
      });
    }

    function renderPagination(totalPages) {
      paginationNode.replaceChildren();

      const pageInfo = document.createElement("p");
      pageInfo.className = "pagination-info";
      pageInfo.textContent = `第 ${state.page} / ${totalPages} 页`;
      paginationNode.appendChild(pageInfo);

      if (totalPages <= 1) {
        return;
      }

      const controls = document.createElement("div");
      controls.className = "pagination-controls";

      controls.appendChild(
        createButton({
          label: "上一页",
          className: "page-button",
          disabled: state.page <= 1,
          onClick: () => {
            state.page -= 1;
            render();
          },
        })
      );

      for (let page = 1; page <= totalPages; page += 1) {
        controls.appendChild(
          createButton({
            label: String(page),
            className: "page-button",
            active: page === state.page,
            onClick: () => {
              state.page = page;
              render();
            },
          })
        );
      }

      controls.appendChild(
        createButton({
          label: "下一页",
          className: "page-button",
          disabled: state.page >= totalPages,
          onClick: () => {
            state.page += 1;
            render();
          },
        })
      );

      paginationNode.appendChild(controls);
    }

    function render() {
      const currentTask = getCurrentTask();
      const filteredVideos = getFilteredVideos(currentTask);
      const totalPages = Math.max(1, Math.ceil(filteredVideos.length / pageSize));
      state.page = Math.min(Math.max(1, state.page), totalPages);

      renderTabs();
      renderStatusFilters();

      const startIndex = (state.page - 1) * pageSize;
      const currentVideos = filteredVideos.slice(startIndex, startIndex + pageSize);
      listNode.replaceChildren();

      if (currentVideos.length === 0) {
        const emptyState = document.createElement("div");
        emptyState.className = "empty-run";
        emptyState.innerHTML = "<p>当前 task 在这个筛选条件下没有视频。</p>";
        listNode.appendChild(emptyState);
      } else {
        currentVideos.forEach((video) => {
          listNode.appendChild(createVideoCard(runPayload, video));
        });
      }

      summaryNode.textContent = `${currentTask.task_label} · ${filteredVideos.length} / ${currentTask.video_count} 个视频 · ${currentTask.success_count} 成功 · ${currentTask.failure_count} 失败`;
      renderPagination(totalPages);
    }

    filterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const nextStatus = button.dataset.statusFilter || "all";
        if (state.status === nextStatus) {
          return;
        }
        state.status = nextStatus;
        state.page = 1;
        render();
      });
    });

    render();
  }

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-open-media]");
    if (!trigger) {
      return;
    }

    openDialog(
      trigger.dataset.openMedia || "video",
      trigger.dataset.src || "",
      trigger.dataset.title || "预览"
    );
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

  document.querySelectorAll("[data-run-column]").forEach((runElement) => {
    initializeRunColumn(runElement);
  });

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
