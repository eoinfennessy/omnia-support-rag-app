export interface Source {
	url: string;
	title: string;
	type: string;
}

export interface ToolCall {
	tool_name: string;
	arguments: Record<string, unknown>;
	result: { data?: unknown; error?: string };
}

export interface ContextChunk {
	id: string;
	page_content: string;
	metadata: { document_id: string; type: string; category: string; [key: string]: unknown };
	score: number;
}

export interface InformationSubQueryResult {
	query: string;
	answer: string;
	context: ContextChunk[];
}

export interface ActionSubQueryResult {
	query: string;
	answer: string;
	tool_calls: ToolCall[];
}

export type SubQueryResult = InformationSubQueryResult | ActionSubQueryResult;

export function isActionResult(r: SubQueryResult): r is ActionSubQueryResult {
	return 'tool_calls' in r;
}

export interface ChatResponse {
	session_id: string;
	answer: string;
	sources: Source[];
	urgency: string | null;
	sub_query_results: SubQueryResult[];
}

export interface Document {
	document_id: string;
	type: string;
	category: string;
	date: string;
	product_category: string;
	content: string;
	resolution_status: string | null;
	customer_satisfaction: number | null;
	priority: string | null;
}

export async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
	const res = await fetch('/api/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ session_id: sessionId, message })
	});
	if (!res.ok) {
		throw new Error(`Chat request failed: ${res.status}`);
	}
	return res.json();
}

export async function clearSession(sessionId: string): Promise<void> {
	await fetch(`/api/chat/${sessionId}`, { method: 'DELETE' });
}

export async function fetchPolicies(): Promise<Document[]> {
	const res = await fetch('/api/policies');
	if (!res.ok) {
		throw new Error(`Failed to fetch policies: ${res.status}`);
	}
	return res.json();
}

export async function fetchFaqs(): Promise<Document[]> {
	const res = await fetch('/api/faqs');
	if (!res.ok) {
		throw new Error(`Failed to fetch FAQs: ${res.status}`);
	}
	return res.json();
}

export async function fetchPolicy(id: string): Promise<Document> {
	const res = await fetch(`/api/policies/${id}`);
	if (!res.ok) {
		throw new Error(`Failed to fetch policy: ${res.status}`);
	}
	return res.json();
}

export async function fetchFaq(id: string): Promise<Document> {
	const res = await fetch(`/api/faqs/${id}`);
	if (!res.ok) {
		throw new Error(`Failed to fetch FAQ: ${res.status}`);
	}
	return res.json();
}
