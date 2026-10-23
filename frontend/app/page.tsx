'use client'

import { useState } from 'react'
import { FileText, MessageSquare, Shield, FlaskConical } from 'lucide-react'
import Chat from '@/components/screens/Chat'
import Documents from '@/components/screens/Documents'
import AttackLab from '@/components/screens/AttackLab'

const nav = [
  { key: 'chat', label: 'Chat', icon: MessageSquare },
  { key: 'documents', label: 'Documents', icon: FileText },
  { key: 'attack-lab', label: 'Attack Lab', icon: FlaskConical },
]

export default function Page() {
  const [active, setActive] = useState('chat')
  const screen =
    active === 'documents' ? <Documents /> :
    active === 'attack-lab' ? <AttackLab /> : <Chat />

  return (
    <div className="min-h-screen bg-[#F4F6F8] text-[#16202A] md:flex">
      <aside className="flex w-full flex-col bg-[#111A22] px-4 py-5 text-[#C9D3DD] md:min-h-screen md:w-[248px] md:shrink-0 md:px-5">
        <div className="mb-8 flex items-center gap-3 px-2 text-white">
          <Shield className="h-7 w-7 text-[#75A7EC]" strokeWidth={1.8} />
          <span className="text-lg font-semibold tracking-tight">RAGShield</span>
        </div>
        <nav className="flex gap-2 md:block md:space-y-2">
          {nav.map(({ key, label, icon: Icon }) => {
            const isActive = active === key
            return (
              <button
                key={key}
                onClick={() => setActive(key)}
                className={`flex flex-1 items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${isActive ? 'bg-[#1F5FBF] text-white' : 'hover:bg-white/10'}`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            )
          })}
        </nav>
        <div className="mt-auto hidden rounded-xl border border-white/10 bg-white/[0.06] p-4 text-sm md:block">
          <p className="font-semibold text-white">Acme Corp</p>
          <p className="mt-1 text-xs text-[#9DAAB6]">Internal helpdesk</p>
        </div>
      </aside>
      <main className="min-w-0 flex-1">{screen}</main>
    </div>
  )
}