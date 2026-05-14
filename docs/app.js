const SECTIONS = [
  'general',
  'therapeutic',
  'surgical',
  'orthopedic',
  'pediatric_therapeutic',
  'pediatric_surgical',
  'orthodontic'
];

const SECTION_LABELS = {
  general: 'Загальний медичний профіль',
  therapeutic: 'Терапевтична стоматологія',
  surgical: 'Хірургічна стоматологія',
  orthopedic: 'Ортопедична стоматологія',
  pediatric_therapeutic: 'Дитяча терапевтична стоматологія',
  pediatric_surgical: 'Дитяча хірургічна стоматологія',
  orthodontic: 'Ортодонтія'
};

const TOTAL_QUESTIONS = 150;
const TIME_LIMIT_SECONDS = 200 * 60;
const PASS_THRESHOLD = 0.64;

function krokApp() {
  return {
    view: 'start',
    loaded: false,
    loadError: null,
    sectionLabels: SECTION_LABELS,

    bank: {},
    bankCounts: {},
    proportions: {},

    questions: [],
    currentIdx: 0,
    answers: {},
    flagged: new Set(),
    showOverview: false,
    showConfirmFinish: false,

    timeLeft: TIME_LIMIT_SECONDS,
    timerInterval: null,
    startedAt: null,
    endedAt: null,

    reviewFilter: 'all',
    result: { correct: 0, percent: 0, passed: false, bySection: {}, timeSpent: 0 },

    get totalBank() {
      return Object.values(this.bankCounts).reduce((a, b) => a + b, 0);
    },

    async init() {
      try {
        const configResp = await fetch('config.json');
        if (!configResp.ok) throw new Error('config.json не знайдено');
        const config = await configResp.json();
        this.proportions = config.proportions || {};

        const bankResults = await Promise.all(
          SECTIONS.map(async (section) => {
            try {
              const resp = await fetch(`questions/${section}.json`);
              if (!resp.ok) return [section, []];
              const data = await resp.json();
              return [section, Array.isArray(data) ? data : (data.questions || [])];
            } catch (e) {
              console.warn(`Failed to load ${section}:`, e);
              return [section, []];
            }
          })
        );

        this.bank = Object.fromEntries(bankResults);
        this.bankCounts = Object.fromEntries(
          bankResults.map(([s, qs]) => [s, qs.length])
        );
        this.loaded = true;
      } catch (e) {
        this.loadError = e.message || String(e);
      }
    },

    sample(arr, n) {
      const copy = [...arr];
      for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
      }
      return copy.slice(0, Math.min(n, copy.length));
    },

    selectQuestions() {
      const targetCounts = {};
      let remaining = TOTAL_QUESTIONS;
      const sortedSections = [...SECTIONS].sort(
        (a, b) => (this.proportions[b] || 0) - (this.proportions[a] || 0)
      );

      for (let i = 0; i < sortedSections.length; i++) {
        const section = sortedSections[i];
        const prop = this.proportions[section] || 0;
        const target = i === sortedSections.length - 1
          ? remaining
          : Math.min(this.bank[section].length, Math.round(TOTAL_QUESTIONS * prop));
        targetCounts[section] = Math.min(target, this.bank[section].length);
        remaining -= targetCounts[section];
      }

      const collected = SECTIONS.flatMap(section =>
        this.sample(this.bank[section], targetCounts[section] || 0)
      );

      if (collected.length < TOTAL_QUESTIONS) {
        const used = new Set(collected.map(q => q.id));
        const pool = SECTIONS
          .flatMap(s => this.bank[s])
          .filter(q => !used.has(q.id));
        const extra = this.sample(pool, TOTAL_QUESTIONS - collected.length);
        collected.push(...extra);
      }

      return this.sample(collected, collected.length);
    },

    startExam() {
      this.questions = this.selectQuestions();
      this.currentIdx = 0;
      this.answers = {};
      this.flagged = new Set();
      this.timeLeft = TIME_LIMIT_SECONDS;
      this.startedAt = Date.now();
      this.endedAt = null;
      this.view = 'exam';
      this.startTimer();
      window.scrollTo(0, 0);
    },

    startTimer() {
      if (this.timerInterval) clearInterval(this.timerInterval);
      this.timerInterval = setInterval(() => {
        this.timeLeft -= 1;
        if (this.timeLeft <= 0) {
          this.timeLeft = 0;
          this.finishExam();
        }
      }, 1000);
    },

    stopTimer() {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
    },

    formatTime(s) {
      const sec = Math.max(0, Math.floor(s));
      const m = Math.floor(sec / 60);
      const r = sec % 60;
      return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`;
    },

    currentQuestion() {
      return this.questions[this.currentIdx];
    },

    nextQuestion() {
      if (this.currentIdx < this.questions.length - 1) {
        this.currentIdx += 1;
        window.scrollTo(0, 0);
      }
    },

    prevQuestion() {
      if (this.currentIdx > 0) {
        this.currentIdx -= 1;
        window.scrollTo(0, 0);
      }
    },

    goTo(idx) {
      this.currentIdx = idx;
      window.scrollTo(0, 0);
    },

    toggleFlag(id) {
      if (this.flagged.has(id)) {
        this.flagged.delete(id);
      } else {
        this.flagged.add(id);
      }
    },

    answeredCount() {
      return Object.keys(this.answers).length;
    },

    confirmFinish() {
      this.showConfirmFinish = true;
    },

    finishExam() {
      this.stopTimer();
      this.endedAt = Date.now();
      this.showConfirmFinish = false;
      this.showOverview = false;

      let correct = 0;
      const bySection = {};
      SECTIONS.forEach(s => bySection[s] = { correct: 0, total: 0 });

      this.questions.forEach(q => {
        if (!bySection[q.section]) bySection[q.section] = { correct: 0, total: 0 };
        bySection[q.section].total += 1;
        if (this.answers[q.id] === q.correct) {
          correct += 1;
          bySection[q.section].correct += 1;
        }
      });

      const percent = (correct / this.questions.length) * 100;
      const timeSpent = Math.floor((this.endedAt - this.startedAt) / 1000);

      this.result = {
        correct,
        percent,
        passed: percent / 100 >= PASS_THRESHOLD,
        bySection,
        timeSpent
      };

      this.view = 'results';
      this.reviewFilter = 'wrong';
      window.scrollTo(0, 0);
    },

    filteredReview() {
      if (this.reviewFilter === 'wrong') {
        return this.questions.filter(q => this.answers[q.id] !== q.correct);
      }
      if (this.reviewFilter === 'flagged') {
        return this.questions.filter(q => this.flagged.has(q.id));
      }
      return this.questions;
    },

    originalIdx(q) {
      return this.questions.indexOf(q);
    },

    restart() {
      this.view = 'start';
      this.questions = [];
      this.answers = {};
      this.flagged = new Set();
      this.timeLeft = TIME_LIMIT_SECONDS;
      this.result = { correct: 0, percent: 0, passed: false, bySection: {}, timeSpent: 0 };
      window.scrollTo(0, 0);
    }
  };
}
