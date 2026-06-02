// Auto-refresh run detail page
function setupAutoRefresh(runId, interval = 3000) {
  const statusBadge = document.querySelector('.run-status');
  if (!statusBadge) return;

  const poll = async () => {
    try {
      const resp = await fetch(`/api/runs/${runId}`);
      if (!resp.ok) return;
      const run = await resp.json();

      const badge = document.querySelector('.run-status');
      if (badge) {
        const oldClass = badge.className.split(' ').find(c => c.includes('status-'));
        const newClass = `status-${run.status}`;
        if (oldClass !== newClass) {
          badge.classList.remove(oldClass);
          badge.classList.add(newClass);
          badge.textContent = run.status;
        }
      }

      // Stop polling if done
      if (run.status !== 'running' && run.status !== 'queued') {
        clearInterval(pollInterval);
      }
    } catch (e) {
      console.error('Poll error:', e);
    }
  };

  const pollInterval = setInterval(poll, interval);
  poll();
}

// Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Show toast notification
function showToast(message, type = 'info') {
  const alertClass = {
    'info': 'alert-info',
    'success': 'alert-success',
    'warning': 'alert-warning',
    'danger': 'alert-danger'
  }[type] || 'alert-info';

  const html = `
    <div class="alert ${alertClass} alert-dismissible fade show position-fixed"
         role="alert" style="top: 20px; right: 20px; z-index: 9999; min-width: 300px;">
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;

  const container = document.body;
  const div = document.createElement('div');
  div.innerHTML = html;
  container.appendChild(div.firstElementChild);
}

// Form handling
document.addEventListener('DOMContentLoaded', () => {
  // Handle form submission with loading state
  const forms = document.querySelectorAll('form[data-loading-text]');
  forms.forEach(form => {
    form.addEventListener('submit', (e) => {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) {
        const originalText = btn.textContent;
        btn.innerHTML = '<span class="spinner me-2"></span>' + btn.dataset.loadingText || 'Loading...';
        btn.disabled = true;
      }
    });
  });

  // Format file sizes
  document.querySelectorAll('[data-bytes]').forEach(el => {
    el.textContent = formatFileSize(parseInt(el.dataset.bytes));
  });
});

// Copy to clipboard
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied to clipboard!', 'success');
  }).catch(() => {
    showToast('Failed to copy', 'danger');
  });
}