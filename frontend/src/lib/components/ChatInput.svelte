<script lang="ts">
	let { onSend, disabled = false }: { onSend: (message: string) => void; disabled?: boolean } =
		$props();
	let input = $state('');

	function handleSubmit() {
		const message = input.trim();
		if (!message || disabled) return;
		input = '';
		onSend(message);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSubmit();
		}
	}
</script>

<form onsubmit={handleSubmit} class="flex gap-3">
	<textarea
		bind:value={input}
		onkeydown={handleKeydown}
		{disabled}
		placeholder="Type your message..."
		rows="1"
		class="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50"
	></textarea>
	<button
		type="submit"
		disabled={disabled || !input.trim()}
		class="rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
	>
		Send
	</button>
</form>
