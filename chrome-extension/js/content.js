// Content script pour extraire les commentaires YouTube

function extractComments() {
  const comments = [];
  
  // Sélectionner tous les éléments de commentaires
  const commentElements = document.querySelectorAll('ytd-comment-thread-renderer');
  
  commentElements.forEach((element) => {
    const commentTextElement = element.querySelector('#content-text');
    if (commentTextElement) {
      const text = commentTextElement.textContent.trim();
      if (text) {
        comments.push(text);
      }
    }
  });
  
  return comments;
}

// Écouter les messages du popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extractComments') {
    const comments = extractComments();
    sendResponse({ comments: comments });
  }
  return true;
});
