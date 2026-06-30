"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { PanelLeftClose, PanelLeftOpen, PenLine, X, ArrowUp, Landmark, User, Gauge } from "lucide-react";

interface Source {
  index: number;
  speaker: string;
  date: string;
  content?: string;
  source_file?: string;
  page?: number;
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

// A real speaker line is short ("Tuan Khoo Poay Tiong [Kota Melaka]"). Some
// rows have a malformed speaker_raw where the whole speech leaked into the
// field (extraction bug); cap it so the card can't balloon.
const MAX_NAME_LEN = 60;

function parseSource(speaker: string) {
  const clean = speaker.replace(/\n/g, " ").trim();
  const match = clean.match(/^(.*?)\s*\[([^\]]+)\]\s*$/);
  if (!match) {
    const name =
      clean.length > MAX_NAME_LEN ? clean.slice(0, MAX_NAME_LEN).trimEnd() + "…" : clean;
    return { name, constituency: "" };
  }

  const outer = match[1].trim();
  const inner = match[2].trim();

  const innerIsName = /\b(bin|binti|bt\.?|Dato'?|Datuk|Datin|Tan Sri|Tun|Dr\b|Tuan|Puan|Haji|Hajah|YB|YAB)\b/i.test(inner);
  const outerIsRole = /\bMenteri\b|\bPengerusi\b|Yang di-Pertua/i.test(outer);

  if (innerIsName || outerIsRole) {
    return { name: inner, constituency: outer };
  }
  return { name: outer, constituency: inner };
}

function buildDisplayMap(text: string): Map<number, number> {
  const seen = new Set<number>();
  const map = new Map<number, number>();
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    const n = parseInt(m[1], 10);
    if (!seen.has(n)) { seen.add(n); map.set(n, map.size + 1); }
  }
  return map;
}

function parseInline(
  text: string,
  msgIdx: number,
  displayMap: Map<number, number>
): React.ReactNode {
  const pattern = /\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[(\d+)\]/g;
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let c = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex)
      nodes.push(<span key={c++}>{text.slice(lastIndex, match.index)}</span>);

    const k = c++;
    if (match[1] !== undefined) {
      nodes.push(<strong key={k}>{match[1]}</strong>);
    } else if (match[2] !== undefined) {
      nodes.push(<em key={k}>{match[2]}</em>);
    } else if (match[3] !== undefined) {
      nodes.push(
        <code key={k} className="bg-gray-100 rounded px-1 py-0.5 text-[0.85em] font-mono text-gray-800">
          {match[3]}
        </code>
      );
    } else if (match[4] !== undefined) {
      const n = parseInt(match[4], 10);
      const displayN = displayMap.get(n) ?? n;
      const targetId = `source-${msgIdx}-${n}`;
      nodes.push(
        <sup key={k} className="text-[10px] leading-none">
          <a
            href={`#${targetId}`}
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(targetId)?.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
            className="text-primary hover:text-primary/70 font-semibold no-underline cursor-pointer"
          >
            [{displayN}]
          </a>
        </sup>
      );
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length)
    nodes.push(<span key={c++}>{text.slice(lastIndex)}</span>);

  return <>{nodes}</>;
}

type MdBlock =
  | { type: "p"; lines: string[] }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "h1" | "h2" | "h3"; content: string }
  | { type: "hr" };

function parseMdBlocks(text: string): MdBlock[] {
  // LLMs often concatenate numbered list items on one line: "1. foo [2]. 2. bar [3]. 3. baz"
  // Split them onto separate lines before parsing.
  const normalized = text
    .split("\n")
    .flatMap((line) => {
      const t = line.trim();
      if (!/^\d+\.\s/.test(t)) return [line];
      return t.replace(/(?<=\S)\.\s+(?=\d+\.\s)/g, ".\n").split("\n");
    })
    .join("\n");

  const blocks: MdBlock[] = [];
  const lines = normalized.split("\n");
  let kind: "p" | "ul" | "ol" | null = null;
  let buf: string[] = [];

  function flush() {
    if (!buf.length) return;
    if (kind === "ul") blocks.push({ type: "ul", items: buf.map(l => l.replace(/^[-*+]\s+/, "")) });
    else if (kind === "ol") blocks.push({ type: "ol", items: buf.map(l => l.replace(/^\d+\.\s+/, "")) });
    else if (kind === "p") {
      while (buf.length && !buf[buf.length - 1].trim()) buf.pop();
      if (buf.length) blocks.push({ type: "p", lines: buf });
    }
    buf = []; kind = null;
  }

  for (const line of lines) {
    const t = line.trim();
    if (!t) { flush(); continue; }
    if (t.startsWith("### ")) { flush(); blocks.push({ type: "h3", content: t.slice(4) }); continue; }
    if (t.startsWith("## "))  { flush(); blocks.push({ type: "h2", content: t.slice(3) }); continue; }
    if (t.startsWith("# "))   { flush(); blocks.push({ type: "h1", content: t.slice(2) }); continue; }
    if (/^[-*]{3,}$/.test(t)) { flush(); blocks.push({ type: "hr" }); continue; }
    if (/^[-*+]\s/.test(t))   { if (kind !== "ul") flush(); kind = "ul"; buf.push(t); continue; }
    if (/^\d+\.\s/.test(t))   { if (kind !== "ol") flush(); kind = "ol"; buf.push(t); continue; }
    if (kind !== "p") flush();
    kind = "p"; buf.push(line);
  }
  flush();
  return blocks;
}

function MarkdownMessage({
  text,
  msgIdx,
  displayMap,
  cursor,
}: {
  text: string;
  msgIdx: number;
  displayMap: Map<number, number>;
  cursor?: boolean;
}) {
  const blocks = parseMdBlocks(text);
  const cursorEl = cursor ? (
    <span className="inline-block w-0.5 h-[0.9em] bg-gray-400 ml-0.5 animate-pulse align-middle" />
  ) : null;

  return (
    <div className="space-y-3 text-base leading-7 text-gray-800">
      {blocks.map((block, bi) => {
        const isLast = bi === blocks.length - 1;
        const k = `b${bi}`;
        if (block.type === "h1")
          return <h2 key={k} className="text-xl font-bold text-gray-900">{parseInline(block.content, msgIdx, displayMap)}{isLast && cursorEl}</h2>;
        if (block.type === "h2")
          return <h3 key={k} className="text-lg font-semibold text-gray-900">{parseInline(block.content, msgIdx, displayMap)}{isLast && cursorEl}</h3>;
        if (block.type === "h3")
          return <h4 key={k} className="text-base font-semibold text-gray-700">{parseInline(block.content, msgIdx, displayMap)}{isLast && cursorEl}</h4>;
        if (block.type === "hr")
          return <hr key={k} className="border-gray-200" />;
        if (block.type === "ul")
          return (
            <ul key={k} className="list-disc pl-5 space-y-1.5">
              {block.items.map((item, ii) => (
                <li key={ii}>
                  {parseInline(item, msgIdx, displayMap)}
                  {isLast && ii === block.items.length - 1 && cursorEl}
                </li>
              ))}
            </ul>
          );
        if (block.type === "ol")
          return (
            <ol key={k} className="list-decimal pl-5 space-y-1.5">
              {block.items.map((item, ii) => (
                <li key={ii}>
                  {parseInline(item, msgIdx, displayMap)}
                  {isLast && ii === block.items.length - 1 && cursorEl}
                </li>
              ))}
            </ol>
          );
        if (block.type === "p")
          return (
            <p key={k}>
              {block.lines.flatMap((line: string, li: number) => {
                const isLastLine = li === block.lines.length - 1;
                const inlined = <span key={`l${li}`}>{parseInline(line, msgIdx, displayMap)}</span>;
                return isLastLine
                  ? [inlined, isLast ? <span key="cursor">{cursorEl}</span> : null]
                  : [inlined, <br key={`br${li}`} />];
              })}
            </p>
          );
        return null;
      })}
      {blocks.length === 0 && cursorEl}
    </div>
  );
}

function SourceCard({
  source,
  id,
  displayIndex,
}: {
  source: Source;
  id: string;
  displayIndex: number;
}) {
  const { name, constituency } = parseSource(source.speaker);
  const [expanded, setExpanded] = useState(false);
  const hasMore = (source.content?.length ?? 0) > 150;

  const docRef = source.source_file
    ? (() => {
        const stem = source.source_file.split("/").pop()?.replace(".pdf", "") ?? "";
        const doc =
          stem.length === 8
            ? `DR ${stem.slice(0, 2)}/${stem.slice(2, 4)}/${stem.slice(4)}`
            : stem;
        return doc + (source.page != null ? ` · p.${source.page}` : "");
      })()
    : null;

  return (
    <Card
      id={id}
      className="border border-gray-100 scroll-mt-4 hover:border-accent/30 hover:shadow-sm transition-all"
    >
      <CardContent className="p-3">
        <div className="flex items-start gap-2.5">
          <Badge className="flex-shrink-0 h-5 w-5 p-0 flex items-center justify-center bg-blue-600 hover:bg-blue-600 text-white text-[10px] font-bold rounded">
            {displayIndex}
          </Badge>
          <div className="min-w-0 flex-1">
            <p className="font-semibold text-gray-900 text-base leading-tight line-clamp-2 break-words">{name}</p>
            {constituency && (
              <p className="text-xs text-gray-400 mt-0.5 leading-tight">{constituency}</p>
            )}
            <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
              <span className="text-xs text-blue-600 font-medium">{source.date}</span>
              {docRef && <span className="text-xs text-gray-400">{docRef}</span>}
            </div>
            {source.content && (
              <>
                <p className="text-xs text-gray-500 mt-2 leading-relaxed break-words whitespace-pre-wrap">
                  {expanded
                    ? source.content
                    : source.content.slice(0, 150) + (hasMore ? "…" : "")}
                </p>
                {hasMore && (
                  <button
                    onClick={() => setExpanded(!expanded)}
                    className="text-primary/80 hover:text-primary text-xs mt-1 font-medium cursor-pointer"
                  >
                    {expanded ? "Show less" : "Show more"}
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function LoadingDots() {
  return (
    <div className="flex gap-1.5 items-center px-1 py-1">
      <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.3s]" />
      <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.15s]" />
      <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce" />
    </div>
  );
}

// Fallback shown only if the backend /suggestions call fails; the live values
// are fetched on mount and reflect real topics, members and dates in the corpus.
const FALLBACK_SUGGESTIONS = [
  "What did members say about fuel subsidies?",
  "Any issues raised about public transport?",
  "What topics were debated in March 2024?",
  "What did Anwar Ibrahim say in parliament?",
];

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

function ChatItem({
  chat, active, renamingId, renameValue, setRenameValue,
  onSelect, onStartRename, onCommitRename, onCancelRename, onDelete,
}: {
  chat: Chat; active: boolean; renamingId: string | null; renameValue: string;
  setRenameValue: (v: string) => void;
  onSelect: () => void; onStartRename: () => void;
  onCommitRename: () => void; onCancelRename: () => void; onDelete: () => void;
}) {
  const isRenaming = renamingId === chat.id;
  return (
    <div
      onClick={onSelect}
      className={`group flex items-center gap-2 rounded-md px-3 py-2 cursor-pointer transition-colors ${
        active ? "bg-white/[0.08] text-white" : "text-slate-400 hover:bg-white/[0.05] hover:text-slate-200"
      }`}
    >
      {isRenaming ? (
        <input
          autoFocus
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onBlur={onCommitRename}
          onKeyDown={(e) => {
            if (e.key === "Enter") onCommitRename();
            if (e.key === "Escape") onCancelRename();
            e.stopPropagation();
          }}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 bg-white/10 text-white text-base rounded px-1.5 py-0.5 outline-none border border-blue-400 min-w-0"
        />
      ) : (
        <span className="truncate flex-1 text-base" onDoubleClick={(e) => { e.stopPropagation(); onStartRename(); }}>
          {chat.title}
        </span>
      )}
      {!isRenaming && (
        <>
          <button onClick={(e) => { e.stopPropagation(); onStartRename(); }}
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-500 hover:text-white flex-shrink-0 transition-opacity cursor-pointer">
            <PenLine className="h-3 w-3" />
          </button>
          <button onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-500 hover:text-white flex-shrink-0 transition-opacity cursor-pointer">
            <X className="h-3 w-3" />
          </button>
        </>
      )}
    </div>
  );
}

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingChatId, setStreamingChatId] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [showAllOpen, setShowAllOpen] = useState(false);
  const [panelExpanded, setPanelExpanded] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>(FALLBACK_SUGGESTIONS);
  const bottomRef = useRef<HTMLDivElement>(null);

  const CHAT_LIMIT = 18;

  function closePanel() {
    setShowAllOpen(false);
    setPanelExpanded(false);
  }

  useEffect(() => {
    const stored = loadChats();
    setChats(stored);
    if (stored.length > 0) setActiveChatId(stored[0].id);
  }, []);

  useEffect(() => {
    // Pull fresh, data-grounded prompts each load; keep the fallback on failure.
    fetch("/api/suggestions")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
          setSuggestions(data.suggestions);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, activeChatId, loading]);

  useEffect(() => {
    // Only collapse sidebar on genuine small screens (phones), not on zoom
    if (window.screen.width < 640) setSidebarOpen(false);
  }, []);

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

  function startRename(chat: Chat) {
    setRenamingId(chat.id);
    setRenameValue(chat.title);
  }

  function commitRename(id: string) {
    const title = renameValue.trim();
    if (title) {
      const updated = chats.map((c) => c.id === id ? { ...c, title } : c);
      setChats(updated);
      saveChats(updated);
    }
    setRenamingId(null);
  }

  async function handleSend(overrideQuery?: string) {
    const query = (overrideQuery ?? input).trim();
    if (!query || loading) return;
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
    setStreamingChatId(currentChatId);
    setStreamingContent("");

    try {
      const res = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "token") {
            accumulated += event.text;
            setStreamingContent(accumulated);
          } else if (event.type === "done") {
            const assistantMsg: Message = {
              role: "assistant",
              content: event.answer,
              sources: event.sources,
            };
            const finalChats = updatedChats.map((c) =>
              c.id === currentChatId
                ? { ...c, messages: [...c.messages, assistantMsg] }
                : c
            );
            setChats(finalChats);
            saveChats(finalChats);
            setStreamingContent("");
          }
        }
      }
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
      setStreamingContent("");
    } finally {
      setLoading(false);
      setStreamingChatId(null);
    }
  }

  return (
    <TooltipProvider>
      <div className="flex h-screen">
        {/* Sidebar */}
        {sidebarOpen && (
          <>
            {/* Mobile backdrop */}
            <div
              className="fixed inset-0 bg-black/50 z-30 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          <aside className="fixed md:relative inset-y-0 left-0 z-40 md:z-auto w-80 flex flex-col flex-shrink-0 bg-navy border-r border-white/[0.06]">
            {/* Brand */}
            <div className="flex items-center justify-between px-4 h-12">
              <div className="flex items-center gap-2 min-w-0">
                <Landmark className="h-4 w-4 text-blue-400 flex-shrink-0" />
                <span className="text-white font-semibold text-base tracking-tight truncate">
                  Hansard Sovereign
                </span>
              </div>
              <Tooltip>
                <TooltipTrigger
                  onClick={newChat}
                  className="h-7 w-7 text-slate-400 hover:text-white hover:bg-white/10 flex-shrink-0 rounded-md flex items-center justify-center transition-colors cursor-pointer"
                >
                  <PenLine className="h-3.5 w-3.5" />
                </TooltipTrigger>
                <TooltipContent side="right" className="text-xs">
                  New chat
                </TooltipContent>
              </Tooltip>
            </div>

            <Separator className="bg-white/[0.06]" />

            {/* Chat list */}
            <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5 scrollbar-dark">
              {chats.length === 0 && (
                <p className="text-center text-xs text-slate-600 mt-6 px-4">
                  Start a new chat to get going
                </p>
              )}
              {chats.slice(0, CHAT_LIMIT).map((chat) => (
                <ChatItem
                  key={chat.id}
                  chat={chat}
                  active={activeChatId === chat.id}
                  renamingId={renamingId}
                  renameValue={renameValue}
                  setRenameValue={setRenameValue}
                  onSelect={() => renamingId !== chat.id && setActiveChatId(chat.id)}
                  onStartRename={() => startRename(chat)}
                  onCommitRename={() => commitRename(chat.id)}
                  onCancelRename={() => setRenamingId(null)}
                  onDelete={() => deleteChat(chat.id)}
                />
              ))}
              {chats.length > CHAT_LIMIT && (
                <button
                  onClick={() => setShowAllOpen(true)}
                  className="w-full text-left px-3 py-2 text-sm text-slate-500 hover:text-slate-300 hover:bg-white/[0.05] rounded-md transition-colors cursor-pointer"
                >
                  Show more ({chats.length - CHAT_LIMIT} more)…
                </button>
              )}
            </div>

            <Separator className="bg-white/[0.06]" />

            <Link
              href="/eval"
              className="flex items-center gap-2 px-4 py-2.5 text-sm text-slate-400 transition-colors hover:bg-white/[0.05] hover:text-slate-200"
            >
              <Gauge className="h-4 w-4 flex-shrink-0 text-blue-400" />
              Inference benchmark
            </Link>

            <Separator className="bg-white/[0.06]" />

            <div className="px-4 py-3">
              <p className="text-xs text-slate-600">
                On-premise · Nothing leaves your machine
              </p>
            </div>
          </aside>
          </>
        )}

        {/* Overflow chats modal */}
        {showAllOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={closePanel}>
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
            <div
              className="relative w-[420px] max-h-[78vh] bg-[#141821] rounded-2xl shadow-2xl border border-white/10 overflow-hidden flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.07]">
                <div>
                  <p className="text-white font-semibold text-base">Chat history</p>
                  <p className="text-slate-500 text-xs mt-0.5">{chats.length - CHAT_LIMIT} older conversations</p>
                </div>
                <button
                  onClick={closePanel}
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* List */}
              <div className={`flex-1 min-h-0 scrollbar-dark ${panelExpanded ? "overflow-y-auto" : "overflow-hidden"}`}>
                {(panelExpanded ? chats.slice(CHAT_LIMIT) : chats.slice(CHAT_LIMIT, CHAT_LIMIT * 2)).map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => { setActiveChatId(chat.id); closePanel(); }}
                    className={`w-full text-left px-5 py-3 border-b border-white/[0.05] last:border-0 transition-colors hover:bg-white/[0.05] ${
                      activeChatId === chat.id ? "bg-white/[0.08]" : ""
                    }`}
                  >
                    <span className="text-slate-200 text-sm truncate block">{chat.title}</span>
                    <span className="text-slate-500 text-xs mt-0.5 block">
                      {new Date(chat.createdAt).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}
                    </span>
                  </button>
                ))}
              </div>

              {/* Inner show more */}
              {!panelExpanded && chats.length > CHAT_LIMIT * 2 && (
                <button
                  onClick={() => setPanelExpanded(true)}
                  className="w-full px-5 py-3 text-sm text-blue-400 hover:text-blue-300 font-medium border-t border-white/[0.07] text-left transition-colors hover:bg-white/[0.03] cursor-pointer"
                >
                  Show all {chats.length - CHAT_LIMIT * 2} more →
                </button>
              )}
            </div>
          </div>
        )}

        {/* Main */}
        <div className="flex-1 flex flex-col bg-slate-50 min-w-0">
          {/* Header */}
          <header className="bg-white border-b border-gray-200 px-4 h-12 flex items-center gap-3 flex-shrink-0">
            <Tooltip>
              <TooltipTrigger
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="h-10 w-10 text-gray-500 hover:text-gray-800 rounded-lg flex items-center justify-center transition-colors cursor-pointer hover:bg-gray-100"
              >
                {sidebarOpen ? <PanelLeftClose className="h-6 w-6" /> : <PanelLeftOpen className="h-6 w-6" />}
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">
                {sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              </TooltipContent>
            </Tooltip>
            <span className="text-base font-medium text-gray-700 truncate">
              {activeChat ? activeChat.title : "Hansard Sovereign"}
            </span>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-4 py-8 space-y-8">
              {!activeChat || activeChat.messages.length === 0 ? (
                /* Empty state */
                <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 shadow-sm">
                    <Landmark className="h-7 w-7 text-primary" />
                  </div>
                  <h1 className="text-2xl font-semibold text-gray-800 mb-2">
                    Hansard Sovereign
                  </h1>
                  <p className="text-sm text-gray-400 mb-8 max-w-sm">
                    Search Malaysian Parliament debates. All processing is on-premise.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md">
                    {suggestions.map((s) => (
                      <button
                        key={s}
                        onClick={() => handleSend(s)}
                        className="bg-white border border-gray-200 rounded-xl p-3 text-sm text-gray-600 hover:border-primary/40 hover:text-primary hover:shadow-sm transition-all text-left leading-snug cursor-pointer"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                /* Message thread */
                activeChat.messages.map((msg, i) => {
                  const displayMap = buildDisplayMap(msg.content);
                  const filteredSources = (
                    displayMap.size > 0
                      ? (msg.sources?.filter((s) => displayMap.has(s.index)) ?? [])
                      : (msg.sources ?? [])
                  ).sort(
                    (a, b) =>
                      (displayMap.get(a.index) ?? a.index) -
                      (displayMap.get(b.index) ?? b.index)
                  );

                  return (
                    <div
                      key={i}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div className="max-w-2xl w-full">
                        {msg.role === "user" ? (
                          <div className="flex justify-end items-start gap-2">
                            <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-3 max-w-lg text-base leading-relaxed">
                              {msg.content}
                            </div>
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-sm">
                              <User className="h-4 w-4 text-white" />
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-sm mt-3">
                              <Landmark className="h-4 w-4 text-white" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <Card className="border border-gray-100 shadow-sm">
                                <CardContent className="px-5 py-4">
                                  <MarkdownMessage
                                    text={msg.content}
                                    msgIdx={i}
                                    displayMap={displayMap}
                                  />
                                </CardContent>
                              </Card>
                              {filteredSources.length > 0 && (
                                <div className="mt-3">
                                  <p className="text-xs text-gray-400 mb-2 ml-1">Sources</p>
                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                    {filteredSources.map((s, si) => (
                                      <SourceCard
                                        key={si}
                                        source={s}
                                        id={`source-${i}-${s.index}`}
                                        displayIndex={displayMap.get(s.index) ?? s.index}
                                      />
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}

              {/* Streaming response */}
              {loading && streamingChatId === activeChatId && (
                <div className="flex items-start gap-3">
                  <div className="avatar-thinking flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-sm mt-3">
                    <Landmark className="h-4 w-4 text-white" />
                  </div>
                  <div className="flex-1 min-w-0 max-w-2xl">
                    <Card className="border border-gray-100 shadow-sm">
                      <CardContent className="px-5 py-4">
                        {streamingContent ? (
                          <MarkdownMessage
                            text={streamingContent}
                            msgIdx={-1}
                            displayMap={new Map()}
                            cursor
                          />
                        ) : (
                          <LoadingDots />
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          {/* Input */}
          <div className="bg-white border-t border-gray-200 px-4 py-4 flex-shrink-0">
            <div className="max-w-3xl mx-auto">
              <div
                className={`flex items-center gap-3 border rounded-2xl px-4 py-3 transition-all ${
                  loading
                    ? "bg-gray-50 border-gray-200"
                    : "bg-white border-gray-200 focus-within:border-primary/60 focus-within:shadow-sm"
                }`}
              >
                <input
                  className="flex-1 text-base outline-none bg-transparent placeholder:text-gray-400 text-gray-800 disabled:cursor-not-allowed"
                  placeholder="Ask about Malaysian Parliament debates…"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                  disabled={loading}
                />
                <Button
                  size="icon"
                  onClick={() => handleSend()}
                  disabled={loading || !input.trim()}
                  className="h-8 w-8 rounded-xl disabled:bg-gray-100 disabled:text-gray-300 flex-shrink-0"
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-center text-xs text-gray-400 mt-2">
                Grounded on Hansard PDFs · Local LLM · On-premise
              </p>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
