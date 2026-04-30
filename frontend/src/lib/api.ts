export interface ChatResponse {
	session_id: string;
	answer: string;
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
