<script lang="ts">
	import type { Source, SubQueryResult } from '$lib/api';
	import { isActionResult } from '$lib/api';
	import { marked } from 'marked';

	let {
		role,
		content,
		sources,
		urgency,
		subQueryResults
	}: {
		role: 'user' | 'assistant';
		content: string;
		sources?: Source[];
		urgency?: string | null;
		subQueryResults?: SubQueryResult[];
	} = $props();

	let htmlContent = $derived(marked.parse(content, { async: false }) as string);

	function pillClasses(type: string): string {
		if (type === 'policy') {
			return 'bg-green-50 text-green-700 ring-green-200 hover:bg-green-100';
		}
		return 'bg-purple-50 text-purple-700 ring-purple-200 hover:bg-purple-100';
	}

	function urgencyClasses(level: string): string {
		switch (level) {
			case 'low':
				return 'bg-green-100 text-green-800';
			case 'medium':
				return 'bg-yellow-100 text-yellow-800';
			case 'high':
				return 'bg-orange-100 text-orange-800';
			case 'critical':
				return 'bg-red-100 text-red-800';
			default:
				return 'bg-gray-100 text-gray-800';
		}
	}

	function prettyPrint(value: unknown): string {
		if (
			typeof value === 'object' &&
			value !== null &&
			'type' in value &&
			(value as Record<string, unknown>).type === 'text' &&
			'text' in value
		) {
			const inner = (value as Record<string, unknown>).text as string;
			try {
				return JSON.stringify(JSON.parse(inner), null, 2);
			} catch {
				return inner;
			}
		}
		return JSON.stringify(value, null, 2);
	}

	let hasDetails = $derived(
		role === 'assistant' && (!!urgency || (subQueryResults && subQueryResults.length > 0))
	);
</script>

<div class="flex {role === 'user' ? 'justify-end' : 'justify-start'}">
	<div
		class="max-w-[80%] rounded-2xl px-4 py-3 {role === 'user'
			? 'bg-blue-600 text-white'
			: 'bg-gray-100 text-gray-900'}"
	>
		<div class="prose prose-sm {role === 'user' ? 'prose-invert' : ''}">
			{@html htmlContent}
		</div>
		{#if hasDetails}
			<details class="mt-2 border-t border-gray-200 pt-2">
				<summary class="cursor-pointer select-none text-xs font-medium text-gray-500">
					Details
				</summary>
				<div class="mt-2 space-y-2">
					{#if urgency}
						<div class="text-xs text-gray-600">
							<span class="font-medium">Urgency:</span>
							<span
								class="ml-1 inline-block rounded-full px-2 py-0.5 font-semibold {urgencyClasses(urgency)}"
							>
								{urgency}
							</span>
						</div>
					{/if}
					{#if subQueryResults && subQueryResults.length > 0}
						<div class="space-y-1.5">
							<span class="text-xs font-medium text-gray-600">Sub-queries:</span>
							{#each subQueryResults as result}
								<details class="rounded-lg bg-white/60">
									<summary
										class="flex cursor-pointer select-none items-start gap-2 px-2.5 py-1.5 text-xs text-gray-700"
									>
										<span
											class="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase
												{isActionResult(result)
												? 'bg-blue-100 text-blue-700'
												: 'bg-emerald-100 text-emerald-700'}"
										>
											{isActionResult(result) ? 'action' : 'info'}
										</span>
										<span>{result.query}</span>
									</summary>
									<div class="space-y-2 px-2.5 pb-2 text-xs text-gray-600">
										<div>
											<span class="font-medium">Answer:</span>
											<p class="mt-0.5 whitespace-pre-wrap">{result.answer}</p>
										</div>
										{#if isActionResult(result) && result.tool_calls.length > 0}
											<div>
												<span class="font-medium">Tool calls:</span>
												{#each result.tool_calls as tc}
													<div class="mt-1 overflow-x-auto rounded bg-gray-50 p-2 font-mono text-[11px]">
														<div class="font-semibold text-gray-800">{tc.tool_name}</div>
														<pre class="mt-0.5 whitespace-pre-wrap text-gray-500">{JSON.stringify(tc.arguments, null, 2)}</pre>
														<div class="mt-0.5">
															{#if tc.result.error}
																<span class="text-red-600">Error: {tc.result.error}</span>
															{:else}
																<pre class="whitespace-pre-wrap text-green-700">{prettyPrint(tc.result.data)}</pre>
															{/if}
														</div>
													</div>
												{/each}
											</div>
										{:else if !isActionResult(result) && result.context.length > 0}
											<div>
												<span class="font-medium">Context documents:</span>
												{#each result.context as chunk}
													<div class="mt-1 rounded bg-gray-50 p-2">
														<div class="flex items-center gap-1.5">
															<span class="font-medium text-gray-800">
																{chunk.metadata.document_id}
															</span>
															<span
																class="rounded-full px-1.5 py-0.5 text-[10px] font-medium
																	{chunk.metadata.type === 'policy'
																	? 'bg-green-50 text-green-700'
																	: chunk.metadata.type === 'faq'
																		? 'bg-purple-50 text-purple-700'
																		: 'bg-gray-100 text-gray-700'}"
															>
																{chunk.metadata.type}
															</span>
															<span class="ml-auto text-[10px] text-gray-400">
																{chunk.score.toFixed(3)}
															</span>
														</div>
														<p class="mt-1 line-clamp-3 whitespace-pre-wrap text-gray-600">
															{chunk.page_content}
														</p>
													</div>
												{/each}
											</div>
										{/if}
									</div>
								</details>
							{/each}
						</div>
					{/if}
				</div>
			</details>
		{/if}
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
