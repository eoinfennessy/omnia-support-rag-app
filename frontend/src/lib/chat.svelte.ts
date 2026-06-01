import type { Source, SubQueryResult } from '$lib/api';

export interface Message {
	role: 'user' | 'assistant';
	content: string;
	sources?: Source[];
	urgency?: string | null;
	subQueryResults?: SubQueryResult[];
}

let messages: Message[] = $state([]);
let sessionId: string = $state('');
let scrollTop: number = 0;

export function getChatState() {
	return {
		get messages() {
			return messages;
		},
		set messages(value: Message[]) {
			messages = value;
		},
		get sessionId() {
			return sessionId;
		},
		set sessionId(value: string) {
			sessionId = value;
		},
		get scrollTop() {
			return scrollTop;
		},
		set scrollTop(value: number) {
			scrollTop = value;
		}
	};
}
