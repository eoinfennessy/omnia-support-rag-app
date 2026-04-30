<script lang="ts">
	let { role, content }: { role: 'user' | 'assistant'; content: string } = $props();
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
	</div>
</div>
