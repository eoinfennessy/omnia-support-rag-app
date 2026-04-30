<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { fetchPolicies, type Document } from '$lib/api';
	import { documentLabel } from '$lib/format';

	let { children } = $props();
	let policies: Document[] = $state([]);
	let loading = $state(true);

	onMount(async () => {
		try {
			policies = await fetchPolicies();
		} finally {
			loading = false;
		}
	});
</script>

<div class="flex h-full">
	<nav class="w-64 shrink-0 overflow-y-auto border-r border-gray-200 bg-gray-50">
		<div class="px-4 py-4">
			<h2 class="text-sm font-semibold text-gray-900">Policies</h2>
		</div>
		{#if loading}
			<p class="px-4 text-sm text-gray-400">Loading...</p>
		{:else}
			<ul>
				{#each policies as doc}
					<li>
						<a
							href="/policies/{doc.document_id}"
							class="block px-4 py-2 text-sm {page.url.pathname === `/policies/${doc.document_id}`
								? 'bg-blue-50 font-medium text-blue-700'
								: 'text-gray-600 hover:bg-gray-100'}"
						>
							{documentLabel(doc)}
						</a>
					</li>
				{/each}
			</ul>
		{/if}
	</nav>
	<div class="flex-1 overflow-y-auto">
		{@render children()}
	</div>
</div>
