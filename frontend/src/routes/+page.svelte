<script lang="ts">
	import { onMount } from 'svelte';
	import { sendMessage, clearSession } from '$lib/api';
	import { getSessionId, resetSession } from '$lib/session';
	import ChatMessage from '$lib/components/ChatMessage.svelte';
	import ChatInput from '$lib/components/ChatInput.svelte';

	interface Message {
		role: 'user' | 'assistant';
		content: string;
	}

	let messages: Message[] = $state([]);
	let loading = $state(false);
	let sessionId = $state('');
	let messagesEnd: HTMLDivElement = $state(undefined!);

	onMount(() => {
		sessionId = getSessionId();
	});

	function scrollToBottom() {
		requestAnimationFrame(() => {
			messagesEnd?.scrollIntoView({ behavior: 'smooth' });
		});
	}

	async function handleSend(text: string) {
		messages.push({ role: 'user', content: text });
		loading = true;
		scrollToBottom();

		try {
			const res = await sendMessage(sessionId, text);
			messages.push({ role: 'assistant', content: res.answer });
		} catch {
			messages.push({
				role: 'assistant',
				content: 'Sorry, something went wrong. Please try again.'
			});
		} finally {
			loading = false;
			scrollToBottom();
		}
	}

	async function handleNewConversation() {
		await clearSession(sessionId);
		sessionId = resetSession();
		messages = [];
	}
</script>

<div class="flex h-screen flex-col bg-white">
	<header class="flex items-center justify-between border-b border-gray-200 px-6 py-4">
		<h1 class="text-lg font-semibold text-gray-900">Omnia Support</h1>
		<button
			onclick={handleNewConversation}
			class="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
		>
			New conversation
		</button>
	</header>

	<main class="flex-1 overflow-y-auto px-6 py-4">
		{#if messages.length === 0}
			<div class="flex h-full items-center justify-center text-gray-400">
				<p>Ask a question about Omnia Retail products, orders, or policies.</p>
			</div>
		{:else}
			<div class="mx-auto flex max-w-3xl flex-col gap-4">
				{#each messages as msg}
					<ChatMessage role={msg.role} content={msg.content} />
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
