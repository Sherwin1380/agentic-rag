import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic RAG — Banking Regulations Assistant",
  description:
    "Hybrid-search RAG over U.S. banking regulations with an agent and an embedding/chunking benchmark.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
