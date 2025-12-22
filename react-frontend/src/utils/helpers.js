export const showToast = (message, type = 'success') => {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  
  // Add toast styles if not already present
  if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `
      .toast {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 300px;
      }
      .toast.success {
        border-left: 4px solid #4caf50;
      }
      .toast.error {
        border-left: 4px solid #ef4444;
      }
      .toast.info {
        border-left: 4px solid #42a5f5;
      }
      @keyframes slideIn {
        from {
          transform: translateX(400px);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
    `;
    document.head.appendChild(style);
  }
  
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
};

export const formatTime = (isoString) => {
  if (!isoString) return 'N/A';
  const date = new Date(isoString);
  return isNaN(date.getTime()) ? 'N/A' : date.toLocaleTimeString();
};

export const formatSpeed = (speed) => {
  return typeof speed === 'number' && !isNaN(speed) ? `${speed.toFixed(1)} km/h` : 'N/A';
};

export const formatDate = (isoString) => {
  if (!isoString) return 'N/A';
  const date = new Date(isoString);
  return isNaN(date.getTime()) ? 'N/A' : date.toLocaleDateString();
};
