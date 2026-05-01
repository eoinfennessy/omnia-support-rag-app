<script lang="ts">
	import { onMount } from 'svelte';
	import { beforeNavigate } from '$app/navigation';
	import { sendMessage, clearSession } from '$lib/api';
	import { getSessionId, resetSession } from '$lib/session';
	import { getChatState } from '$lib/chat.svelte';
	import ChatMessage from '$lib/components/ChatMessage.svelte';
	import ChatInput from '$lib/components/ChatInput.svelte';

	const chat = getChatState();

	let loading = $state(false);
	let messagesEnd: HTMLDivElement = $state(undefined!);
	let mainEl: HTMLElement = $state(undefined!);

	onMount(() => {
		if (!chat.sessionId) {
			chat.sessionId = getSessionId();
		}
		if (chat.scrollTop && mainEl) {
			mainEl.scrollTop = chat.scrollTop;
		}
	});

	beforeNavigate(() => {
		if (mainEl) {
			chat.scrollTop = mainEl.scrollTop;
		}
	});

	function scrollToBottom() {
		requestAnimationFrame(() => {
			messagesEnd?.scrollIntoView({ behavior: 'smooth' });
		});
	}

	async function handleSend(text: string) {
		chat.messages.push({ role: 'user', content: text });
		loading = true;
		scrollToBottom();

		try {
			const res = await sendMessage(chat.sessionId, text);
			chat.messages.push({ role: 'assistant', content: res.answer, sources: res.sources });
		} catch {
			chat.messages.push({
				role: 'assistant',
				content: 'Sorry, something went wrong. Please try again.'
			});
		} finally {
			loading = false;
			scrollToBottom();
		}
	}

	async function handleNewConversation() {
		await clearSession(chat.sessionId);
		chat.sessionId = resetSession();
		chat.messages = [];
	}
</script>

<div class="flex h-full flex-col bg-white">
	<header class="flex items-center justify-between border-b border-gray-200 px-6 py-4">
		<h1 class="text-lg font-semibold text-gray-900">Omnia Support</h1>
		<button
			onclick={handleNewConversation}
			class="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
		>
			New conversation
		</button>
	</header>

	<main bind:this={mainEl} class="flex-1 overflow-y-auto px-6 py-4">
		{#if chat.messages.length === 0}
			<div class="flex h-full items-center justify-center text-gray-400">
				<p>Ask a question about Omnia Retail products, orders, or policies.</p>
			</div>
		{:else}
			<div class="mx-auto flex max-w-3xl flex-col gap-4">
				{#each chat.messages as msg}
					<ChatMessage role={msg.role} content={msg.content} sources={msg.sources} />
				{/each}
				{#if loading}
					<div class="flex justify-start">
						<div class="rounded-2xl bg-gray-100 px-4 py-3 text-sm text-gray-500">
							Thinking...
						</div>
					</div>
				{/if}
				<div bind:this={messagesEnd}></div>
			</div>
		{/if}
	</main>

	<footer class="border-t border-gray-200 px-6 py-4">
		<div class="mx-auto max-w-3xl">
			<ChatInput onSend={handleSend} disabled={loading} />
		</div>
	</footer>
</div>
