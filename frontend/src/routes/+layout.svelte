<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<div class="flex h-screen flex-col">
	<nav class="flex border-b border-gray-200 bg-white px-6">
		<a
			href="/"
			class="border-b-2 px-4 py-3 text-sm font-medium {page.url.pathname === '/'
				? 'border-blue-600 text-blue-600'
				: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'}"
		>
			Chat
		</a>
		<a
			href="/policies"
			class="border-b-2 px-4 py-3 text-sm font-medium {page.url.pathname.startsWith('/policies')
				? 'border-blue-600 text-blue-600'
				: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'}"
		>
			Policies
		</a>
		<a
			href="/faqs"
			class="border-b-2 px-4 py-3 text-sm font-medium {page.url.pathname.startsWith('/faqs')
				? 'border-blue-600 text-blue-600'
				: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'}"
		>
			FAQs
		</a>
	</nav>
	<div class="flex-1 overflow-hidden">
		{@render children()}
	</div>
</div>
