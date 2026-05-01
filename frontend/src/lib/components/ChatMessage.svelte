<script lang="ts">
	import type { Source } from '$lib/api';

	let {
		role,
		content,
		sources
	}: { role: 'user' | 'assistant'; content: string; sources?: Source[] } = $props();

	function pillClasses(type: string): string {
		if (type === 'policy') {
			return 'bg-green-50 text-green-700 ring-green-200 hover:bg-green-100';
		}
		return 'bg-purple-50 text-purple-700 ring-purple-200 hover:bg-purple-100';
	}
</script>

<div class="flex {role === 'user' ? 'justify-end' : 'justify-start'}">
	<div
		class="max-w-[80%] rounded-2xl px-4 py-3 {role === 'user'
			? 'bg-blue-600 text-white'
			: 'bg-gray-100 text-gray-900'}"
	>
		<div class="prose prose-sm {role === 'user' ? 'prose-invert' : ''}">
			{@html content}
		</div>
		{#if sources && sources.length > 0}
			<div class="mt-2 flex flex-wrap items-center gap-2 border-t border-gray-200 pt-2">
				<span class="text-xs font-medium text-gray-500">Sources:</span>
				{#each sources as source}
					<a
						href={source.url}
						class="rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 {pillClasses(source.type)}"
					>
						{source.title}
					</a>
				{/each}
			</div>
		{/if}
	</div>
</div>
