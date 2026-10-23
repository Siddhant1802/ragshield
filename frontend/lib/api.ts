export const API_BASE = 'http://localhost:8000'

export type Defense = { defenses: string[]; default: string }
export type TraceStage = { stage: 'input' | 'retrieval' | 'output'; status: 'passed' | 'blocked' | 'modified' | 'skipped'; details: string[] }
export type AskResponse = { answer: string; sources: string[]; blocked: { source: string; section: string; reason: string }[]; trace: TraceStage[] }
export type Document = { name: string; trust: 'trusted' | 'untrusted'; chunks: number; origin: 'built-in' | 'uploaded' }
export type Result = { label: string; attack_runs: number; at_risk: number; at_risk_pct: number; at_risk_ci: [number, number]; full_success: number; poison_blocked: number; normal_correct: number; normal_total: number; false_positives: number; avg_latency_s: number; cost_per_1000: number }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) throw new Error(`Request failed (${response.status})`)
  return response.json()
}
export const getDefenses = () => request<Defense>('/api/defenses')
export const ask = (question: string, defense: string) => request<AskResponse>('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, defense }) })
export const getDocuments = () => request<{ documents: Document[] }>('/api/documents')
export const uploadDocument = (file: File, trust: Document['trust']) => { const body = new FormData(); body.append('file', file); body.append('trust', trust); return request('/api/documents', { method: 'POST', body }) }
export const deleteDocument = (name: string) => request(`/api/documents/${encodeURIComponent(name)}`, { method: 'DELETE' })
export const getResults = () => request<{ results: Result[] }>('/api/results')
