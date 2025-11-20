<script lang="ts">
    import './chatbotcss.css';
    import { onMount } from "svelte";
    import { loadChatData, getGreeting, processUserMessage, isReady, type Message } from "./chatbot"

    let messages: Message[] = [];
    let chatInput = "";
    let isTyping = false;
    let chatMessagesDiv: HTMLElement;
    let botReady = false;

    onMount(async () => {
        try {
            await loadChatData();
            
            if (isReady()) {
                botReady = true;
                const greeting = getGreeting();
                messages = [{ sender: 'bot', text: greeting }];
            } else {
                messages = [{ sender: 'bot', text: 'Initializing AI systems...' }];
            }
        } catch (err) {
            console.error("Failed to load chatbot:", err);
            messages = [{ sender: 'bot', text: 'Failed to initialize. Please refresh the page.' }];
        }
    });

    $: if (messages.length && chatMessagesDiv) {
        setTimeout(() => {
            chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
        }, 50);
    }

    async function sendMessage() {
        if (!chatInput.trim() || isTyping || !botReady) { 
            if (!botReady) {
                console.log("Bot not ready yet!");
            }
            return; 
        }

        const userMsg = chatInput.trim();
        chatInput = "";
        
        messages = [...messages, { sender: 'user', text: userMsg }];

        isTyping = true;
        
        setTimeout(async () => {
            const botResponse = await processUserMessage(userMsg);
            messages = [...messages, { sender: 'bot', text: botResponse }];
            isTyping = false;
        }, 800 + Math.random() * 1200);
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }
</script>

<section class="chatbot-section">
    <div class="chatbot-wrapper">
        <div class="chatbot-header">
            <h2 class="chatbot-title">AFIRA AI ASSISTANT</h2>
            <span class="version">v0.0.8</span>
            {#if !botReady}
                <span class="loading-badge">INITIALIZING...</span>
            {/if}
        </div>

        <div class="chat-container">
            <div class="chat-messages" bind:this={chatMessagesDiv}>
                {#each messages as message}
                    <div class="message {message.sender}">
                        <span class="sender-tag">[{message.sender === 'bot' ? 'AFIRA' : 'YOU'}]</span>
                        <span class="message-text">{message.text}</span>
                    </div>
                {/each}
                
                {#if isTyping}
                    <div class="message bot">
                        <span class="sender-tag">[AFIRA]</span>
                        <span class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </span>
                    </div>
                {/if}
            </div>

            <div class="input-wrapper">
                <span class="input-prefix">></span>
                <input 
                    type="text"
                    class="chat-input"
                    placeholder={botReady ? "Type your message..." : "Initializing..."}
                    bind:value={chatInput}
                    on:keydown={handleKeydown}
                    disabled={!botReady}
                />
            </div>
        </div>

        <div class="chatbot-footer">
            <div class="footer-line"></div>
            <p class="footer-text">Powered by Str1k3rFl0 Neural Network • Experimental AI System</p>
        </div>
    </div>
</section>