   let currentVideoData = null;
      let selectedFormat = null;
      let currentTaskId = null;
      let progressInterval = null;

      // DOM elements
      const urlForm = document.getElementById("urlForm");
      const urlInput = document.getElementById("urlInput");
      const extractBtn = document.getElementById("extractBtn");
      const loading = document.getElementById("loading");
      const error = document.getElementById("error");
      const success = document.getElementById("success");
      const videoInfo = document.getElementById("videoInfo");
      const thumbnail = document.getElementById("thumbnail");
      const videoTitle = document.getElementById("videoTitle");
      const videoDuration = document.getElementById("videoDuration");
      const videoUploader = document.getElementById("videoUploader");
      const videoViews = document.getElementById("videoViews");
      const videoLikes = document.getElementById("videoLikes");
      const uploadDate = document.getElementById("uploadDate");
      const videoDescription = document.getElementById("videoDescription");
      const formatsGrid = document.getElementById("formatsGrid");
      const downloadBtn = document.getElementById("downloadBtn");
      const downloadSection = document.getElementById("downloadSection");
      const taskStatus = document.getElementById("taskStatus");
      const taskMessage = document.getElementById("taskMessage");
      const progressFill = document.getElementById("progressFill");
      const progressText = document.getElementById("progressText");
      const downloadComplete = document.getElementById("downloadComplete");
      const downloadLink = document.getElementById("downloadLink");
      const playBtn = document.getElementById("playBtn");
      const videoPlayer = document.getElementById("videoPlayer");
      const videoElement = document.getElementById("videoElement");

      // Utility functions
      function showError(message) {
        error.textContent = message;
        error.style.display = "block";
        success.style.display = "none";
        setTimeout(() => {
          error.style.display = "none";
        }, 5000);
      }

      function showSuccess(message) {
        success.textContent = message;
        success.style.display = "block";
        error.style.display = "none";
        setTimeout(() => {
          success.style.display = "none";
        }, 3000);
      }

      function hideAllSections() {
        loading.style.display = "none";
        videoInfo.style.display = "none";
        downloadSection.style.display = "none";
        videoPlayer.style.display = "none";
      }

      function formatFileSize(bytes) {
        if (!bytes) return "Unknown size";
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (
          Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i]
        );
      }

      function formatDuration(seconds) {
        if (!seconds) return "Unknown duration";
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
          return `${hours}:${mins.toString().padStart(2, "0")}:${secs
            .toString()
            .padStart(2, "0")}`;
        }
        return `${mins}:${secs.toString().padStart(2, "0")}`;
      }

      function formatNumber(num) {
        if (!num) return "Unknown";
        return num.toLocaleString();
      }

      function formatDate(dateStr) {
        if (!dateStr) return "Unknown date";
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        return `${day}/${month}/${year}`;
      }

      // Extract video information
      urlForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();
        if (!url) {
          showError("Please enter a valid YouTube URL");
          return;
        }

        hideAllSections();
        loading.style.display = "block";
        extractBtn.disabled = true;

        try {
          const response = await fetch("/extract", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: url }),
          });

          const data = await response.json();

          if (response.ok && data.success) {
            currentVideoData = data.data;
            currentVideoData.url = url;
            displayVideoInfo(data.data);
            showSuccess("Video information extracted successfully!");
          } else {
            showError(
              data.error || data.detail || "Failed to extract video information"
            );
          }
        } catch (err) {
          console.error("Network error:", err);
          showError(
            "Network error. Please check your connection and try again."
          );
        } finally {
          loading.style.display = "none";
          extractBtn.disabled = false;
        }
      });

      function displayVideoInfo(data) {
        // Set video details
        thumbnail.src =
          data.thumbnail ||
          "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='150' viewBox='0 0 200 150'%3E%3Crect width='200' height='150' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' font-size='14' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Thumbnail%3C/text%3E%3C/svg%3E";
        thumbnail.onerror = function () {
          this.src =
            "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='150' viewBox='0 0 200 150'%3E%3Crect width='200' height='150' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' font-size='14' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Thumbnail%3C/text%3E%3C/svg%3E";
        };

        videoTitle.textContent = data.title || "YouTube Video";
        videoDuration.textContent = `⏱️ Duration: ${formatDuration(
          data.duration
        )}`;
        videoUploader.textContent = `📺 Channel: ${data.uploader || "Unknown"}`;
        videoViews.textContent = `👁️ ${formatNumber(data.view_count)} views`;
        videoLikes.textContent = `👍 ${formatNumber(data.like_count)} likes`;
        uploadDate.textContent = `📅 ${formatDate(data.upload_date)}`;
        videoDescription.textContent =
          data.description || "No description available";

        // Display formats
        formatsGrid.innerHTML = "";

        if (!data.formats || data.formats.length === 0) {
          formatsGrid.innerHTML =
            '<p style="color: #666; text-align: center; padding: 20px;">No formats available for this video.</p>';
          return;
        }

        data.formats.forEach((format, index) => {
          const formatCard = document.createElement("div");
          formatCard.className = "format-card";
          formatCard.dataset.formatId = format.format_id;
          formatCard.dataset.quality = format.quality;

          if (index === 0) {
            formatCard.classList.add("recommended");
          }

          if (format.ext === "mp3" || format.quality.includes("Audio Only")) {
            formatCard.classList.add("audio-only");
          }

          let qualityText = format.quality || "Unknown Quality";
          let formatDetails = "";

          if (format.ext === "mp3" || format.quality.includes("Audio Only")) {
            formatDetails = `
                        Format: ${format.ext?.toUpperCase() || "MP3"}<br>
                        Quality: 320kbps<br>
                        Audio: ${format.acodec || "MP3"} ✅<br>
                        <strong style="color: #6f42c1;">🎵 Perfect for Music</strong>
                    `;
          } else {
            formatDetails = `
                        Format: ${format.ext?.toUpperCase() || "MP4"}<br>
                        Resolution: ${format.width}x${
              format.height || "Auto"
            }<br>
                        ${format.fps ? `FPS: ${format.fps}<br>` : ""}
                        Video: ${format.vcodec || "H.264"}<br>
                        Audio: ${format.acodec || "AAC"} ✅
                        ${
                          index === 0
                            ? '<br><strong style="color: #28a745;">⭐ BEST QUALITY</strong>'
                            : ""
                        }
                    `;
          }

          formatCard.innerHTML = `
                    <div class="format-quality">${qualityText}</div>
                    <div class="format-details">
                        ${formatDetails}
                    </div>
                `;

          formatCard.addEventListener("click", () => {
            document.querySelectorAll(".format-card").forEach((card) => {
              card.classList.remove("selected");
            });

            formatCard.classList.add("selected");
            selectedFormat = format;
            downloadBtn.style.display = "block";
          });

          formatsGrid.appendChild(formatCard);
        });

        // Auto-select first format
        if (data.formats.length > 0) {
          const firstCard = formatsGrid.firstElementChild;
          firstCard.classList.add("selected");
          selectedFormat = data.formats[0];
          downloadBtn.style.display = "block";
        }

        videoInfo.style.display = "block";
      }

      // Download video with async task tracking
      downloadBtn.addEventListener("click", async () => {
        if (!selectedFormat || !currentVideoData) {
          showError("Please select a format first");
          return;
        }

        downloadSection.style.display = "block";
        downloadComplete.style.display = "none";
        downloadBtn.disabled = true;

        // Reset progress
        updateProgress(0, "Starting download...", "processing");

        try {
          const response = await fetch("/download", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              url: currentVideoData.url,
              format_id: selectedFormat.format_id,
              quality: selectedFormat.quality,
            }),
          });

          const data = await response.json();

          if (response.ok && data.success) {
            currentTaskId = data.task_id;
            showSuccess(`Download started! Task ID: ${data.task_id}`);

            // Start progress tracking
            startProgressTracking();
          } else {
            showError(data.error || data.detail || "Failed to start download");
            downloadSection.style.display = "none";
          }
        } catch (err) {
          console.error("Download error:", err);
          showError(
            "Failed to start download. Please check your connection and try again."
          );
          downloadSection.style.display = "none";
        } finally {
          downloadBtn.disabled = false;
        }
      });

      function updateProgress(progress, message, status) {
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
        taskMessage.textContent = message;

        // Update task status styling
        taskStatus.className = `task-status ${status}`;
      }

      function startProgressTracking() {
        if (progressInterval) {
          clearInterval(progressInterval);
        }

        progressInterval = setInterval(async () => {
          if (!currentTaskId) return;

          try {
            const response = await fetch(`/task/${currentTaskId}`);
            const data = await response.json();

            if (response.ok && data.success) {
              const task = data.task;

              updateProgress(
                task.progress || 0,
                task.message || "Processing...",
                task.status
              );

              if (task.status === "completed") {
                clearInterval(progressInterval);
                handleDownloadComplete(task);
              } else if (task.status === "failed") {
                clearInterval(progressInterval);
                handleDownloadFailed(task);
              }
            }
          } catch (err) {
            console.error("Progress tracking error:", err);
          }
        }, 1000); // Check every second
      }

      function handleDownloadComplete(task) {
        updateProgress(100, "Download completed successfully!", "completed");

        setTimeout(() => {
          downloadComplete.style.display = "block";

          // Set download link
          downloadLink.href = task.download_url;
          downloadLink.download = task.filename;

          // Show play button for video files
          if (
            selectedFormat.ext !== "mp3" &&
            !selectedFormat.quality.includes("Audio Only")
          ) {
            videoElement.src = task.download_url;
            playBtn.style.display = "inline-block";
          } else {
            playBtn.style.display = "none";
          }

          const fileType = selectedFormat.ext === "mp3" ? "audio" : "video";
          showSuccess(`High quality ${fileType} downloaded successfully!`);
        }, 1000);
      }

      function handleDownloadFailed(task) {
        updateProgress(0, task.error || "Download failed", "failed");
        showError(task.error || "Download failed. Please try again.");

        setTimeout(() => {
          downloadSection.style.display = "none";
        }, 3000);
      }

      // Play video
      playBtn.addEventListener("click", () => {
        videoPlayer.style.display = "block";
        videoElement.scrollIntoView({ behavior: "smooth" });
      });

      // Cleanup on page unload
      window.addEventListener("beforeunload", () => {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
      });

      // Auto-focus URL input
      urlInput.focus();