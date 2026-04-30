<script lang="ts">
	import type { Document } from '$lib/api';
	import { documentLabel } from '$lib/format';

	let { document }: { document: Document } = $props();
</script>

<article class="rounded-lg border border-gray-200 bg-white p-6">
	<h2 class="mb-3 text-lg font-semibold text-gray-900">{documentLabel(document)}</h2>
	<div class="mb-3 flex flex-wrap items-center gap-2">
		{#if document.product_category !== 'all' && document.product_category !== 'none'}
			<span class="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600">
				{document.product_category}
			</span>
		{/if}
		<span class="ml-auto text-xs text-gray-400">{document.date}</span>
	</div>
	<div class="prose prose-sm max-w-none whitespace-pre-line text-gray-700">
		{document.content}
	</div>
</article>
