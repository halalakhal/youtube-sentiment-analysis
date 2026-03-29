let allResults = [];
let currentFilter = 'all';

// Configuration API - URL de votre API Hugging Face
let API_URL = 'https://halalakhal-youtube-sentiment-api.hf.space';

// Charger la configuration au d√©marrage
document.addEventListener('DOMContentLoaded', () => {
  // Charger l'URL de l'API sauvegard√©e
  chrome.storage.local.get(['apiUrl', 'theme'], (result) => {
    if (result.apiUrl) {
      API_URL = result.apiUrl;
      document.getElementById('apiUrl').value = API_URL;
    }
    if (result.theme) {
      document.documentElement.setAttribute('data-theme', result.theme);
    }
  });

  // Event listeners
  document.getElementById('analyzeBtn').addEventListener('click', analyzeComments);
  document.getElementById('saveApi').addEventListener('click', saveApiUrl);
  document.getElementById('themeToggle').addEventListener('click', toggleTheme);
  document.getElementById('copyResults').addEventListener('click', copyResults);
  document.getElementById('exportCSV').addEventListener('click', exportCSV);

  // Filtres
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      currentFilter = e.target.dataset.filter;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      displayResults(allResults);
    });
  });
});

function saveApiUrl() {
  const url = document.getElementById('apiUrl').value.trim();
  API_URL = url;
  chrome.storage.local.set({ apiUrl: url }, () => {
    showTemporaryMessage('API URL saved!');
  });
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  chrome.storage.local.set({ theme: newTheme });
}

async function analyzeComments() {
  hideError();
  showLoading();
  hideResults();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab.url.includes('youtube.com/watch')) {
      throw new Error('Please open a YouTube video page');
    }

    const response = await chrome.tabs.sendMessage(tab.id, { action: 'extractComments' });
    const comments = response.comments;

    if (!comments || comments.length === 0) {
      throw new Error('No comments found. Please scroll down to load comments first.');
    }

    console.log(`Extracted ${comments.length} comments`);

    const apiResponse = await fetch(`${API_URL}/predict_batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ comments: comments })
    });

    if (!apiResponse.ok) {
      throw new Error(`API error: ${apiResponse.status}`);
    }

    const data = await apiResponse.json();
    console.log('API response:', data);

    allResults = data.results;
    displayStatistics(data.statistics);
    displayResults(data.results);
    showResults();

  } catch (error) {
    console.error('Error:', error);
    showError(error.message);
  } finally {
    hideLoading();
  }
}

function displayStatistics(stats) {
  document.getElementById('positiveCount').textContent = stats.positive;
  document.getElementById('neutralCount').textContent = stats.neutral;
  document.getElementById('negativeCount').textContent = stats.negative;
  document.getElementById('positivePercent').textContent = `${stats.positive_percentage}%`;
  document.getElementById('neutralPercent').textContent = `${stats.neutral_percentage}%`;
  document.getElementById('negativePercent').textContent = `${stats.negative_percentage}%`;
  document.getElementById('totalCount').textContent = stats.total_comments;
  document.getElementById('avgConfidence').textContent = `${(stats.average_confidence * 100).toFixed(1)}%`;
}

function displayResults(results) {
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = '';

  let filteredResults = results;
  if (currentFilter !== 'all') {
    filteredResults = results.filter(r => r.sentiment === parseInt(currentFilter));
  }

  if (filteredResults.length === 0) {
    resultsContainer.innerHTML = '<p style="text-align: center; color: #5f6368;">No comments match the filter</p>';
    return;
  }

  filteredResults.forEach(result => {
    const card = document.createElement('div');
    card.className = `comment-card ${result.sentiment_label.toLowerCase()}`;
    
    const emoji = result.sentiment === 1 ? 'Ì∏ä' : result.sentiment === 0 ? 'Ì∏ê' : 'Ì∏û';
    
    card.innerHTML = `
      <div class="comment-header">
        <span class="sentiment-badge ${result.sentiment_label.toLowerCase()}">
          ${emoji} ${result.sentiment_label}
        </span>
        <span class="confidence">${(result.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="comment-text">${escapeHtml(result.text)}</div>
    `;
    
    resultsContainer.appendChild(card);
  });
}

function copyResults() {
  const text = allResults.map(r => 
    `[${r.sentiment_label}] (${(r.confidence * 100).toFixed(1)}%) ${r.text}`
  ).join('\n\n');
  
  navigator.clipboard.writeText(text).then(() => {
    showTemporaryMessage('Results copied to clipboard!');
  });
}

function exportCSV() {
  const csv = [
    ['Text', 'Sentiment', 'Confidence'].join(','),
    ...allResults.map(r => [
      `"${r.text.replace(/"/g, '""')}"`,
      r.sentiment_label,
      r.confidence
    ].join(','))
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sentiment-analysis-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function showLoading() {
  document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
  const errorDiv = document.getElementById('error');
  errorDiv.textContent = message;
  errorDiv.classList.remove('hidden');
}

function hideError() {
  document.getElementById('error').classList.add('hidden');
}

function showResults() {
  document.getElementById('statistics').classList.remove('hidden');
  document.getElementById('filters').classList.remove('hidden');
  document.getElementById('results').classList.remove('hidden');
  document.getElementById('actions').classList.remove('hidden');
}

function hideResults() {
  document.getElementById('statistics').classList.add('hidden');
  document.getElementById('filters').classList.add('hidden');
  document.getElementById('results').classList.add('hidden');
  document.getElementById('actions').classList.add('hidden');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showTemporaryMessage(message) {
  const originalText = document.getElementById('analyzeBtn').querySelector('.btn-text').textContent;
  document.getElementById('analyzeBtn').querySelector('.btn-text').textContent = message;
  setTimeout(() => {
    document.getElementById('analyzeBtn').querySelector('.btn-text').textContent = originalText;
  }, 2000);
}
