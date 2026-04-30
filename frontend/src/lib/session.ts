const SESSION_KEY = 'omnia-session-id';

function generateId(): string {
	return crypto.randomUUID();
}

export function getSessionId(): string {
	let id = localStorage.getItem(SESSION_KEY);
	if (!id) {
		id = generateId();
		localStorage.setItem(SESSION_KEY, id);
	}
	return id;
}

export function resetSession(): string {
	const id = generateId();
	localStorage.setItem(SESSION_KEY, id);
	return id;
}
