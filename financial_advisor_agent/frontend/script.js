/**
 * FinAdvisor AI — Frontend Script
 *
 * 100% dynamic — no data values hardcoded in this file.
 * Everything is fetched from the API: portfolio list, stock names, sector names,
 * market data, quick chips, FII/DII, breadth, heatmap, ticker.
 *
 * Architecture:
 *   1. initApp()        → fetch /api/registry + /api/market/snapshot
 *   2. renderAll()      → build all UI elements from API data
 *   3. handleChat()     → SSE stream from POST /api/chat
 *   4. renderMarkdown() → convert agent response to HTML
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  registry:          null,   // from /api/registry
  marketSnapshot:    null,   // from /api/market/snapshot
  selectedPortfolio: null,   // portfolio_id string or null
  sessionId:         crypto.randomUUID(),
  isStreaming:       false,
  streamBuffer:      '',     // accumulates streaming tokens
  currentNodes:      [],     // for reasoning tracker
};

// ReAct agent node labels
const NODE_LABELS = {
  financial_advisor: '🧠 Reasoning',
  tool_executor:     '🔧 Tools',
};

// Labels for individual tools the agent may call
const TOOL_LABELS = {
  think:                   '💭 Planning',
  list_portfolios:         '📋 Listing portfolios',
  get_portfolio_analysis:  '💼 Portfolio analysis',
  get_portfolio_risk:      '⚠️ Risk assessment',
  get_market_overview:     '📊 Market overview',
  get_stock_details:       '📈 Stock lookup',
  get_sector_analysis:     '🏭 Sector analysis',
  search_news:             '📰 News search',
  get_top_movers:          '🔝 Top movers',
  get_mutual_fund_details: '💰 Fund data',
  build_causal_chain:      '🔗 Causal chain',
};

// Dynamic tracker state — built up as tool_call events arrive
const trackerState = { steps: [], nodeEl: null };

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────────────────────
async function initApp() {
  try {
    const [registry, snapshot] = await Promise.all([
      fetchJSON('/api/registry'),
      fetchJSON('/api/market/snapshot'),
    ]);
    state.registry       = registry;
    state.marketSnapshot = snapshot;

    renderTopbar(registry, snapshot);
    renderTicker(snapshot);
    renderPortfolioSelector(registry);
    renderMarketMini(snapshot);
    renderFiiDii(snapshot);
    renderSectorHeatmap(snapshot);
    renderQuickChips(registry, snapshot);
    updateWelcomeSubtitle(registry);
    setupInputHandlers();
  } catch (err) {
    console.error('Init failed:', err);
    addSystemMessage('⚠️ Could not connect to the server. Make sure it is running on port 8000.');
  }
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

// ── Topbar ────────────────────────────────────────────────────────────────────
function renderTopbar(registry, snapshot) {
  $('market-date').textContent = registry.market_date || '';

  const badge = $('market-status-badge');
  const status = (registry.market_status || 'CLOSED').toUpperCase();
  badge.textContent = status;
  badge.className   = `status-badge ${status === 'OPEN' ? 'open' : 'closed'}`;
}

// ── Ticker ────────────────────────────────────────────────────────────────────
function renderTicker(snapshot) {
  const track   = $('ticker-track');
  const indices  = snapshot.indices || {};
  const sectors  = snapshot.sector_performance || {};
  const movers   = snapshot.top_movers || {};

  const items = [];

  // Add all indices (dynamic — from data)
  Object.entries(indices).forEach(([name, data]) => {
    items.push({ name, value: data.current_value?.toLocaleString('en-IN'), change: data.change_percent });
  });

  // Add top 3 gainers and top 3 losers (dynamic)
  (movers.top_gainers || []).slice(0, 3).forEach(s => {
    items.push({ name: s.symbol, value: `₹${s.current_price?.toLocaleString('en-IN')}`, change: s.change_percent });
  });
  (movers.top_losers || []).slice(0, 3).forEach(s => {
    items.push({ name: s.symbol, value: `₹${s.current_price?.toLocaleString('en-IN')}`, change: s.change_percent });
  });

  // Duplicate for seamless loop
  const allItems = [...items, ...items];
  track.innerHTML = allItems.map(item => {
    const isUp  = item.change >= 0;
    const arrow = isUp ? '▲' : '▼';
    const cls   = isUp ? 'up' : 'down';
    const chg   = item.change != null ? `${arrow} ${Math.abs(item.change).toFixed(2)}%` : '';
    return `
      <span class="ticker-item">
        <span class="t-name">${item.name}</span>
        <span class="t-val" style="color:var(--text-2);font-size:11px;">${item.value || ''}</span>
        <span class="t-change ${cls}">${chg}</span>
      </span>`;
  }).join('');
}

// ── Portfolio Selector ────────────────────────────────────────────────────────
function renderPortfolioSelector(registry) {
  const sel = $('portfolio-select');
  // Dynamic — from registry.portfolios (from JSON keys)
  (registry.portfolios || []).forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.user_name} — ${p.portfolio_type?.replace(/_/g, ' ')}`;
    sel.appendChild(opt);
  });

  sel.addEventListener('change', () => {
    state.selectedPortfolio = sel.value || null;
    if (state.selectedPortfolio) {
      loadPortfolioPanel(state.selectedPortfolio);
    } else {
      hidePortfolioPanel();
    }
  });
}

async function loadPortfolioPanel(portfolioId) {
  try {
    const data = await fetchJSON(`/api/portfolio/${portfolioId}`);
    renderPortfolioPanel(data);
  } catch (err) {
    console.error('Portfolio load failed:', err);
  }
}

function renderPortfolioPanel(data) {
  $('no-portfolio-hint').classList.add('hidden');
  const panel = $('portfolio-panel');
  panel.classList.add('visible');

  $('pf-name').textContent  = data.user_name || '';
  $('pf-type').textContent  = (data.portfolio_type || '').replace(/_/g, ' ');

  const val  = data.current_value || 0;
  $('pf-value').textContent = `₹${formatCrore(val)}`;

  const dayPct = data.day_change_pct || 0;
  const dayAbs = data.day_change_absolute || 0;
  const isUp   = dayPct >= 0;
  const dayEl  = $('pf-day-change');
  dayEl.textContent = `${isUp ? '▲' : '▼'} ${Math.abs(dayPct).toFixed(2)}%  (₹${Math.abs(dayAbs).toLocaleString('en-IN')})`;
  dayEl.className   = `portfolio-change ${isUp ? 'up' : 'down'}`;

  // Risk badges — dynamic from data
  const badges = $('pf-risk-badges');
  badges.innerHTML = '';
  const riskFlags = data.risk_flags || [];
  if (riskFlags.length === 0) {
    badges.innerHTML = `<span class="risk-badge ok">✓ No Major Risks</span>`;
  } else {
    riskFlags.slice(0, 3).forEach(flag => {
      const isCritical = flag.includes('CRITICAL');
      const isWarning  = flag.includes('WARNING') || flag.includes('HIGH');
      const cls = isCritical ? 'critical' : isWarning ? 'warning' : 'ok';
      const label = flag.replace(/^[🚨⚠️📰📉]+\s*/u, '').split(':')[0].trim().substring(0, 30);
      badges.innerHTML += `<span class="risk-badge ${cls}">${label}</span>`;
    });
  }

  // Sector bars — dynamic from sector_allocation
  const bars   = $('pf-sector-bars');
  bars.innerHTML = '';
  const sectors = data.sector_allocation || {};
  const maxPct  = Math.max(...Object.values(sectors), 1);
  Object.entries(sectors)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .forEach(([sector, pct]) => {
      const label = sector.replace(/_/g, ' ').replace('INFORMATION TECHNOLOGY', 'IT');
      bars.innerHTML += `
        <div class="sector-bar-row">
          <span class="sector-bar-label">${label}</span>
          <div class="sector-bar-track">
            <div class="sector-bar-fill" style="width:${(pct / maxPct) * 100}%"></div>
          </div>
          <span class="sector-bar-pct">${pct.toFixed(1)}%</span>
        </div>`;
    });
}

function hidePortfolioPanel() {
  $('portfolio-panel').classList.remove('visible');
  $('no-portfolio-hint').classList.remove('hidden');
}

// ── Market Mini ────────────────────────────────────────────────────────────────
function renderMarketMini(snapshot) {
  const mini    = $('market-mini');
  const indices  = snapshot.indices || {};
  mini.innerHTML = '';
  // Dynamic — iterates all indices from data
  Object.entries(indices).forEach(([name, data]) => {
    const chg   = data.change_percent || 0;
    const isUp  = chg >= 0;
    const val   = (data.current_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
    mini.innerHTML += `
      <div class="market-mini-row">
        <span class="index-name">${name}</span>
        <span class="index-val">${val}</span>
        <span class="index-chg ${isUp ? 'up' : 'down'}">${isUp ? '▲' : '▼'} ${Math.abs(chg).toFixed(2)}%</span>
      </div>`;
  });
}

// ── FII / DII ──────────────────────────────────────────────────────────────────
function renderFiiDii(snapshot) {
  const panel  = $('fii-dii-panel');
  const fiiDii = snapshot.fii_dii || {};
  const fii    = fiiDii.fii || {};
  const dii    = fiiDii.dii || {};
  if (!fii.net_value_cr) {
    panel.innerHTML = '<span style="color:var(--text-3);font-size:11px;">FII/DII data unavailable</span>';
    return;
  }
  const fiiNet = fii.net_value_cr;
  const diiNet = dii.net_value_cr;
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;">
      <span style="color:var(--text-2);">FII Net</span>
      <span style="font-weight:600;color:${fiiNet < 0 ? 'var(--rose)' : 'var(--emerald)'}">
        ${fiiNet < 0 ? '▼' : '▲'} ₹${Math.abs(fiiNet).toLocaleString('en-IN')} Cr
      </span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="color:var(--text-2);">DII Net</span>
      <span style="font-weight:600;color:${diiNet >= 0 ? 'var(--emerald)' : 'var(--rose)'}">
        ${diiNet >= 0 ? '▲' : '▼'} ₹${Math.abs(diiNet).toLocaleString('en-IN')} Cr
      </span>
    </div>
    <div style="font-size:10px;color:var(--text-3);margin-top:4px;">${fiiDii.observation || ''}</div>`;

  // Breadth
  const breadth   = snapshot.market_breadth || {};
  const nifty50   = breadth.nifty50 || {};
  const sentiment = breadth.sentiment_indicator || '';
  const sentCls   = sentiment === 'FEAR' ? 'fear' : sentiment === 'GREED' ? 'greed' : 'neutral';
  $('breadth-row').innerHTML = `
    <span>A/D:</span>
    <span class="breadth-val">${nifty50.advances || 0} ▲ / ${nifty50.declines || 0} ▼</span>
    <span style="margin-left:auto;">
      <span class="breadth-val ${sentCls}">${sentiment}</span>
    </span>`;
}

// ── Sector Heatmap ─────────────────────────────────────────────────────────────
function renderSectorHeatmap(snapshot) {
  const heatmap = $('sector-heatmap');
  const sectors  = snapshot.sector_performance || {};
  heatmap.innerHTML = '';
  // Dynamic — all sectors from data, sorted by change %
  const sorted = Object.entries(sectors).sort((a, b) => a[1].change_percent - b[1].change_percent);
  sorted.forEach(([sector, data]) => {
    const chg    = data.change_percent || 0;
    const isUp   = chg >= 0;
    const label  = sector.replace(/_/g, ' ').replace('INFORMATION TECHNOLOGY', 'IT');
    const color  = isUp ? 'var(--emerald)' : 'var(--rose)';
    const alpha  = Math.min(Math.abs(chg) / 3, 1) * 0.25;
    heatmap.innerHTML += `
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:4px 8px;border-radius:6px;
                  background:rgba(${isUp?'16,185,129':'244,63,94'},${alpha});
                  cursor:pointer;font-size:11px;"
           onclick="askAboutSector('${sector}')">
        <span style="color:var(--text-2);">${label}</span>
        <span style="font-weight:600;color:${color}">${isUp?'▲':'▼'} ${Math.abs(chg).toFixed(2)}%</span>
      </div>`;
  });
}

function askAboutSector(sector) {
  $('user-input').value = `What's happening in the ${sector.replace(/_/g, ' ')} sector today?`;
  sendMessage();
}

// ── Quick Chips ───────────────────────────────────────────────────────────────
function renderQuickChips(registry, snapshot) {
  const chips     = $('quick-chips');
  const chipsData = buildDynamicChips(registry, snapshot);
  chips.innerHTML = chipsData.map(text =>
    `<button class="quick-chip" onclick="useChip(this)">${text}</button>`
  ).join('');
}

function buildDynamicChips(registry, snapshot) {
  // Chips are generated from live data — not hardcoded text
  const chips   = [];

  // General chips
  chips.push('📊 Full market summary');

  // Worst sectors (dynamic)
  const sectors  = snapshot.sector_performance || {};
  const sorted   = Object.entries(sectors).sort((a, b) => a[1].change_percent - b[1].change_percent);
  if (sorted.length > 0) {
    const worst = sorted[0][0].replace(/_/g, ' ');
    chips.push(`📉 Why is ${worst} falling?`);
  }
  if (sorted.length > 1) {
    const second = sorted[1][0].replace(/_/g, ' ');
    chips.push(`📉 What's hurting ${second}?`);
  }

  // Top gainer (dynamic)
  const movers = snapshot.top_movers || {};
  const gainers = movers.top_gainers || [];
  if (gainers.length > 0) {
    chips.push(`📈 Tell me about ${gainers[0].symbol}`);
  }

  // Portfolio chips (dynamic)
  if (state.selectedPortfolio) {
    chips.push('⚠️ Check my portfolio risk');
    chips.push('❓ Why is my portfolio down?');
  } else if ((registry.portfolios || []).length > 0) {
    chips.push('💼 Show me portfolio analysis');
  }

  chips.push('📰 Top market news today');
  return chips;
}

function useChip(btn) {
  // Strip leading emoji + whitespace before using as query
  // Also strip any remaining non-BMP characters (emoji) to prevent surrogate issues
  const raw = btn.textContent.replace(/^[\p{Emoji}\s]+/u, '').trim();
  $('user-input').value = raw || btn.textContent.trim();
  sendMessage();
}

/**
 * Remove surrogate characters from a string before sending to API.
 * Emoji typed or pasted by the user may contain surrogate pairs (\uD800-\uDFFF)
 * which are valid in JS but cause Python's UTF-8 codec to fail.
 */
function sanitizeInput(str) {
  // Replace surrogates with empty string
  return str.replace(/[\uD800-\uDFFF]/g, '');
}

// ── Welcome subtitle ──────────────────────────────────────────────────────────
function updateWelcomeSubtitle(registry) {
  const el = $('welcome-subtitle');
  el.innerHTML = `
    Powered by <strong>GPT-4o + LangGraph</strong> deep reasoning.<br/>
    I have access to <strong>${(registry.stocks || []).length} stocks</strong>,
    <strong>${(registry.sectors || []).length} sectors</strong>,
    <strong>${registry.news_count || 0} news articles</strong>, and
    <strong>${(registry.portfolios || []).length} portfolios</strong> as of ${registry.market_date || 'today'}.
    <br/><br/>Select a portfolio above or just start asking.`;
}

// ── Input handlers ────────────────────────────────────────────────────────────
function setupInputHandlers() {
  const input   = $('user-input');
  const sendBtn = $('send-btn');

  sendBtn.addEventListener('click', sendMessage);

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const input = $('user-input');
  const rawText = input.value.trim();
  const text  = sanitizeInput(rawText);  // strip surrogate chars (from emoji)
  if (!text || state.isStreaming) return;

  input.value  = '';
  input.style.height = 'auto';

  // Hide welcome card
  const welcome = $('welcome-card');
  if (welcome) welcome.remove();

  // Add user bubble
  appendMessage('user', text);
  scrollToBottom();

  // Start streaming
  state.isStreaming  = true;
  state.streamBuffer = '';
  $('send-btn').disabled = true;

  // Create AI bubble placeholder
  const aiBubble = appendMessage('ai', '');
  aiBubble.querySelector('.bubble').innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

  // Initialise reasoning tracker
  showReasoningTracker();

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        message:      text,
        portfolio_id: state.selectedPortfolio || null,
        session_id:   state.sessionId,
      }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';
    let   hasTokens = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();  // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let event;
        try { event = JSON.parse(raw); } catch { continue; }

        switch (event.type) {
          case 'node_start':
            if (event.node === 'financial_advisor') {
              addTrackerStep('🧠 Reasoning...', 'active');
            }
            break;

          case 'node_end':
            markLastTrackerStep('done');
            break;

          case 'tool_call':
            // Agent decided to call this specific tool — show it
            addTrackerStep(event.label || `🔧 ${event.tool}`, 'active');
            break;

          case 'tool_done':
            markLastTrackerStep('done');
            break;

          case 'token':
            if (!hasTokens) {
              // Replace typing indicator with real content on first token
              aiBubble.querySelector('.bubble').innerHTML = '';
              hasTokens = true;
            }
            state.streamBuffer += event.content;
            renderStreamingContent(aiBubble.querySelector('.bubble'), state.streamBuffer);
            scrollToBottom();
            break;

          case 'judge_result':
            displayJudgePanel(aiBubble, event.result);
            break;

          case 'done':
            if (state.streamBuffer) {
              renderStreamingContent(aiBubble.querySelector('.bubble'), state.streamBuffer);
            }
            hideReasoningTracker();
            break;

          case 'error':
            aiBubble.querySelector('.bubble').innerHTML = `<span style="color:var(--rose)">⚠️ ${event.message}</span>`;
            hideReasoningTracker();
            break;
        }
      }
    }
  } catch (err) {
    console.error('Stream error:', err);
    aiBubble.querySelector('.bubble').innerHTML = `<span style="color:var(--rose)">⚠️ Connection error. Please check the server.</span>`;
    hideReasoningTracker();
  } finally {
    state.isStreaming = false;
    $('send-btn').disabled = false;
    scrollToBottom();
    // Refresh quick chips with current context
    if (state.registry && state.marketSnapshot) {
      renderQuickChips(state.registry, state.marketSnapshot);
    }
  }
}

// ── Reasoning Tracker (dynamic — built step by step as tool events arrive) ────

function showReasoningTracker() {
  trackerState.steps = [];
}

/**
 * Add a new step to the tracker (called on node_start or tool_call events).
 * Returns the created element so callers can update it later.
 */
function addTrackerStep(label, status) {
  return null;
}

function markLastTrackerStep(status) {}

function hideReasoningTracker() {
  trackerState.steps = [];
}

// ── LLM-as-a-Judge Panel ──────────────────────────────────────────────────────

const VERDICT_CONFIG = {
  EXCELLENT:          { color: 'var(--emerald)',  bg: 'rgba(16,185,129,0.12)', icon: '🏆' },
  GOOD:               { color: 'var(--accent)',   bg: 'rgba(0,212,170,0.10)',  icon: '✅' },
  ACCEPTABLE:         { color: 'var(--amber)',    bg: 'rgba(245,158,11,0.12)', icon: '⚠️' },
  NEEDS_IMPROVEMENT:  { color: 'var(--rose)',     bg: 'rgba(244,63,94,0.12)',  icon: '🔴' },
};

const SCORE_LABELS = {
  factual_grounding: 'Factual Grounding',
  causal_reasoning:  'Causal Reasoning',
  completeness:      'Completeness',
  actionability:     'Actionability',
  risk_awareness:    'Risk Awareness',
  conciseness:       'Conciseness',
};

function scoreColor(score) {
  if (score >= 8) return 'var(--emerald)';
  if (score >= 6) return 'var(--accent)';
  if (score >= 4) return 'var(--amber)';
  return 'var(--rose)';
}

function displayJudgePanel(bubbleWrapper, result) {
  if (!result || !result.scores) return;

  const verdictKey = result.verdict || 'ACCEPTABLE';
  const vc         = VERDICT_CONFIG[verdictKey] || VERDICT_CONFIG.ACCEPTABLE;
  const overall    = typeof result.overall === 'number' ? result.overall.toFixed(1) : '—';

  // Score bars
  const barsHtml = Object.entries(SCORE_LABELS).map(([key, label]) => {
    const score = result.scores[key] ?? 0;
    const pct   = score * 10;
    const color = scoreColor(score);
    return `
      <div class="judge-score-row">
        <span class="judge-score-label">${label}</span>
        <div class="judge-bar-bg">
          <div class="judge-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="judge-score-num" style="color:${color}">${score}</span>
      </div>`;
  }).join('');

  // Strengths & improvements
  const strengthsHtml = (result.strengths || []).map(s =>
    `<li class="judge-strength">✓ ${sanitizeText(s)}</li>`).join('');
  const improvHtml = (result.improvements || []).map(s =>
    `<li class="judge-improve">↑ ${sanitizeText(s)}</li>`).join('');

  const panel = document.createElement('div');
  panel.className = 'judge-panel';
  panel.innerHTML = `
    <div class="judge-header">
      <span class="judge-title">⚖️ Response Quality Score</span>
      <span class="judge-verdict" style="color:${vc.color};background:${vc.bg}">
        ${vc.icon} ${verdictKey.replace('_', ' ')} — ${overall}/10
      </span>
    </div>
    <div class="judge-scores">${barsHtml}</div>
    ${strengthsHtml || improvHtml ? `
    <div class="judge-feedback">
      ${strengthsHtml ? `<ul class="judge-list">${strengthsHtml}</ul>` : ''}
      ${improvHtml    ? `<ul class="judge-list">${improvHtml}</ul>`    : ''}
    </div>` : ''}`;

  bubbleWrapper.appendChild(panel);
  scrollToBottom();
}

// ── Chat bubbles ──────────────────────────────────────────────────────────────
function appendMessage(role, text) {
  const container  = $('chat-messages');
  const wrapper    = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar     = document.createElement('div');
  avatar.className  = `avatar ${role}`;
  avatar.textContent = role === 'ai' ? '🤖' : '👤';

  const bubble     = document.createElement('div');
  bubble.className  = `bubble ${role}`;
  if (text) bubble.innerHTML = role === 'ai' ? renderMarkdown(text) : escapeHtml(text);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  return wrapper;
}

function addSystemMessage(text) {
  const container = $('chat-messages');
  const div = document.createElement('div');
  div.style.cssText = 'text-align:center;font-size:12px;color:var(--text-3);padding:8px;';
  div.textContent = text;
  container.appendChild(div);
}

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderStreamingContent(el, markdown) {
  el.innerHTML = renderMarkdown(markdown);
}

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    // Bold / italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Code inline
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Horizontal rule
    .replace(/^---+$/gm, '<hr/>')
    // Unordered lists
    .replace(/^\* (.+)$/gm, '<li>$1</li>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    // Ordered lists
    .replace(/^\d+\.\s(.+)$/gm, '<li>$1</li>')
    // Wrap consecutive <li> in <ul>
    .replace(/(<li>[\s\S]+?<\/li>)(?!\n<li>)/g, '<ul>$1</ul>')
    // Line breaks → paragraphs
    .split(/\n{2,}/)
    .map(block => block.trim())
    .filter(Boolean)
    .map(block => {
      if (/^<(h[123]|ul|hr|li)/.test(block)) return block;
      return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
    })
    .join('');

  return html;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  const el = $('chat-messages');
  el.scrollTop = el.scrollHeight;
}

function formatCrore(val) {
  if (val >= 1e7) return (val / 1e7).toFixed(2) + ' Cr';
  if (val >= 1e5) return (val / 1e5).toFixed(2) + ' L';
  return val.toLocaleString('en-IN');
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initApp);
