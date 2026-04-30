export interface ChatResponse {
	session_id: string;
	answer: string;
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
