"use client";

import { useState, useRef, useEffect } from "react";

interface Source {
  speaker: string;
  date: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

function parseSource(speaker: string) {
  const match = speaker.match(/^(.*?)\s*\[([^\]]+)\]\s*$/);
  if (match) return { name: match[1].trim(), constituency: match[2].trim() };
  return { name: speaker, constituency: "" };
}

function SourceCard({ source }: { source: Source }) {
  const { name, constituency } = parseSource(source.speaker);
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 text-sm">
      <p className="font-medium text-gray-800">{name}</p>
      {constituency && <p className="text-gray-500 text-xs">{constituency}</p>}
      <p className="text-blue-500 text-xs mt-1">{source.date}</p>
    </div>
  );
}

function LoadingDots() {
  return (
    <div className="flex gap-1 items-center p-4">
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
    </div>
  );
}

const STORAGE_KEY = "hansard_chats";

function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveChats(chats: Chat[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = loadChats();
    setChats(stored);
    if (stored.length > 0) setActiveChatId(stored[0].id);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, activeChatId, loading]);

  const activeChat = chats.find((c) => c.id === activeChatId) ?? null;

  function newChat() {
    const chat: Chat = {
      id: Date.now().toString(),
      title: "New chat",
      messages: [],
      createdAt: Date.now(),
    };
    const updated = [chat, ...chats];
    setChats(updated);
    saveChats(updated);
    setActiveChatId(chat.id);
  }

  function deleteChat(id: string) {
    const updated = chats.filter((c) => c.id !== id);
    setChats(updated);
    saveChats(updated);
    if (activeChatId === id) {
      setActiveChatId(updated.length > 0 ? updated[0].id : null);
    }
  }

  async function handleSend() {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");

    let currentChatId = activeChatId;
    let currentChats = chats;

    if (!currentChatId) {
      const chat: Chat = {
        id: Date.now().toString(),
        title: query.slice(0, 40),
        messages: [],
        createdAt: Date.now(),
      };
      currentChats = [chat, ...chats];
      setChats(currentChats);
      saveChats(currentChats);
      setActiveChatId(chat.id);
      currentChatId = chat.id;
    }

    const userMsg: Message = { role: "user", content: query };
    const updatedChats = currentChats.map((c) =>
      c.id === currentChatId
        ? {
            ...c,
            title: c.messages.length === 0 ? query.slice(0, 40) : c.title,
            messages: [...c.messages, userMsg],
          }
        : c
    );
    setChats(updatedChats);
    saveChats(updatedChats);
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      const finalChats = updatedChats.map((c) =>
        c.id === currentChatId
          ? { ...c, messages: [...c.messages, assistantMsg] }
          : c
      );
      setChats(finalChats);
      saveChats(finalChats);
    } catch {
      const errorMsg: Message = {
        role: "assistant",
        content: "Something went wrong. Please try again.",
      };
      const finalChats = updatedChats.map((c) =>
        c.id === currentChatId
          ? { ...c, messages: [...c.messages, errorMsg] }
          : c
      );
      setChats(finalChats);
      saveChats(finalChats);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900">
      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-64 bg-gray-900 text-white flex flex-col">
          <div className="p-4 border-b border-gray-700">
            <button
              onClick={newChat}
              className="w-full bg-gray-700 hover:bg-gray-600 rounded-lg px-4 py-2 text-sm text-left transition-colors"
            >
              + New chat
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`group flex items-center justify-between rounded-lg px-3 py-2 cursor-pointer text-sm transition-colors ${
                  activeChatId === chat.id
                    ? "bg-gray-700"
                    : "hover:bg-gray-800"
                }`}
                onClick={() => setActiveChatId(chat.id)}
              >
                <span className="truncate">{chat.title}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-white ml-2 text-xs"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
            Hansard Sovereign · Local LLM
          </div>
        </div>
      )}

      {/* Main */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b bg-white px-4 py-3 flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-gray-500 hover:text-gray-800"
          >
            ☰
          </button>
          <span className="font-medium">
            {activeChat ? activeChat.title : "Hansard Sovereign"}
          </span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 max-w-3xl w-full mx-auto">
          {!activeChat || activeChat.messages.length === 0 ? (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-xl font-medium text-gray-600">Hansard Sovereign</p>
              <p className="text-sm mt-2">Malaysian Parliament debates · On-premise · Nothing leaves your machine</p>
              <div className="mt-8 grid grid-cols-2 gap-3 max-w-md mx-auto text-left">
                {[
                  "What did members say about fuel subsidies?",
                  "Any issues raised about public transport?",
                  "What topics were debated in March 2024?",
                  "What did Anwar Ibrahim say in parliament?",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => {
                      setInput(suggestion);
                    }}
                    className="bg-white border border-gray-200 rounded-lg p-3 text-sm text-gray-600 hover:border-blue-400 hover:text-blue-600 transition-colors text-left"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            activeChat.messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className="max-w-2xl w-full">
                  {msg.role === "user" ? (
                    <div className="flex justify-end">
                      <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-lg">
                        {msg.content}
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 whitespace-pre-wrap text-gray-800">
                        {msg.content}
                      </div>
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-400 mb-2">Sources</p>
                          <div className="grid grid-cols-2 gap-2">
                            {msg.sources.map((s, j) => (
                              <SourceCard key={j} source={s} />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm">
                <LoadingDots />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t bg-white px-4 py-4">
          <div className="max-w-3xl mx-auto flex gap-2">
            <input
              className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 text-sm"
              placeholder="Ask about Malaysian Parliament debates..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={loading}
            />
            <button
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl px-5 py-3 transition-colors text-sm"
              onClick={handleSend}
              disabled={loading}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}