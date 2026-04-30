<script lang="ts">
	import { page } from '$app/state';
	import { fetchPolicy, type Document } from '$lib/api';
	import DocumentCard from '$lib/components/DocumentCard.svelte';

	let document: Document | null = $state(null);
	let loading = $state(true);
	let error = $state('');

	$effect(() => {
		const id = page.params.id;
		if (!id) {
			error = 'Policy not found.';
			loading = false;
			return;
		}
		loading = true;
		error = '';
		document = null;
		fetchPolicy(id)
			.then((doc) => (document = doc))
			.catch(() => (error = 'Policy not found.'))
			.finally(() => (loading = false));
	});
</script>

<div class="px-6 py-4">
	{#if loading}
		<p class="text-gray-400">Loading...</p>
	{:else if error}
		<p class="text-red-500">{error}</p>
	{:else if document}
		<div class="mx-auto max-w-3xl">
			<DocumentCard {document} />
		</div>
	{/if}
</div>
