import type { Document } from '$lib/api';

export function toTitleCase(snake: string): string {
	return snake
		.split('_')
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(' ');
}

export function documentLabel(doc: Document): string {
	if (doc.type === 'faq') {
		return `${doc.document_id} - ${toTitleCase(doc.category)}`;
	}
	return toTitleCase(doc.category);
}
