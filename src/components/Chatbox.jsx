import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { SendHorizonal } from "lucide-react";
import { Leapfrog } from "ldrs/react";
import { motion, AnimatePresence } from "framer-motion";

const Chatbox = ({ parsedContent, messages, setMessages }) => {
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesContainerRef = useRef(null);
  const latestUserMessageRef = useRef(null);

  const backendEndpoint =
    import.meta.env.VITE_BACKEND_API_URL || "http://localhost:5000";

  useEffect(() => {
    if (latestUserMessageRef.current && messagesContainerRef.current) {
      const containerRect =
        messagesContainerRef.current.getBoundingClientRect();
      const messageRect = latestUserMessageRef.current.getBoundingClientRect();

      const scrollPosition = messageRect.top - containerRect.top - 24;

      messagesContainerRef.current.scrollBy({
        top: scrollPosition,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!userInput.trim()) return;

    const newMessage = { role: "user", content: userInput.trim() };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setUserInput("");
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(
        `${backendEndpoint}/api/chat`,
        {
          messages: updatedMessages,
          parsedContent: parsedContent,
        },
        {
          headers: {
            "Content-Type": "application/json",
            skip_zrok_interstitial: true,
          },
        }
      );

      const assistantMessage = response.data;
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Error sending chat message:", err);
      setError(
        "Sorry, I encountered an error processing your request. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  // Variants for animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
      },
    },
  };

  // Message bubble animations
  const bubbleVariants = {
    hidden: {
      opacity: 0,
      y: 20,
      scale: 0.95,
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 500,
        damping: 40,
      },
    },
  };

  // Message pop-up animation
  const popUpVariants = {
    hidden: {
      opacity: 0,
      scale: 0.8,
      y: 10,
    },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 30,
      },
    },
  };

  // Error animation
  const errorVariants = {
    hidden: { opacity: 0, y: -10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 500,
        damping: 30,
      },
    },
    exit: {
      opacity: 0,
      y: -10,
      transition: { duration: 0.2 },
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="w-full max-w-2xl mx-auto border border-gray-300 rounded-lg shadow-sm overflow-hidden bg-white"
    >
      <div className="border-t border-gray-300"></div>

      {/* Messages container */}
      <motion.div
        ref={messagesContainerRef}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="md:h-[455px] xl:h-[600px] h-96 overflow-y-auto p-4 space-y-4 bg-gray-50"
      >
        <AnimatePresence>
          {messages.map((msg, index) => {
            const isLatestUserMessage =
              msg.role === "user" &&
              index === messages.findLastIndex((m) => m.role === "user");

            return (
              <motion.div
                key={index}
                ref={isLatestUserMessage ? latestUserMessageRef : null}
                variants={
                  index === messages.length - 1 ? popUpVariants : bubbleVariants
                }
                initial="hidden"
                animate="visible"
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <motion.div
                  className={`p-3 max-w-[80%] ${
                    msg.role === "user"
                      ? "bg-blue-100 rounded-l-xl rounded-tr-xl"
                      : "bg-gray-200 rounded-r-xl rounded-tl-xl"
                  }`}
                >
                  <div className="prose prose-sm">
                    <ReactMarkdown
                      components={{
                        // Proper paragraph handling for newlines
                        p: ({ children }) => (
                          <p className="mb-4 last:mb-0">{children}</p>
                        ),

                        // Bold text
                        strong: ({ children }) => (
                          <span className="font-bold">{children}</span>
                        ),

                        // Handle numbered lists
                        ol: ({ children }) => (
                          <ol className="list-decimal pl-5 mb-4 space-y-1">
                            {children}
                          </ol>
                        ),

                        // Handle bullet lists
                        ul: ({ children }) => (
                          <ul className="list-disc pl-5 mb-4 space-y-1">
                            {children}
                          </ul>
                        ),

                        // List items
                        li: ({ children }) => (
                          <li className="mb-1">{children}</li>
                        ),
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                </motion.div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* AI Loading indicator */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}
              className="flex justify-start"
            >
              <motion.div
                animate={{
                  scale: [1, 1.03, 1],
                  transition: { repeat: Infinity, duration: 2 },
                }}
                className="bg-gray-200 px-3 pb-1 pt-3 rounded-r-xl rounded-tl-xl"
              >
                <div>
                  <Leapfrog size="30" color="gray"></Leapfrog>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error message */}
        <AnimatePresence>
          {error && (
            <motion.div
              variants={errorVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              className="bg-red-100 text-red-800 p-3 rounded-lg"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Input area */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.3 }}
        className="border-t border-gray-300 p-3 bg-white"
      >
        <div className="flex items-center">
          <motion.input
            transition={{ duration: 0.2 }}
            type="text"
            className="flex-1 rounded-l-md p-2"
            placeholder="Ask about your career matches..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
            disabled={loading}
          />
          <motion.button
            whileTap={{ scale: 0.95 }}
            transition={{ duration: 0.2 }}
            onClick={handleSend}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-[11px] rounded-r-md hover:bg-blue-700 disabled:bg-blue-400 focus:outline-none"
          >
            <SendHorizonal size={20} />
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Chatbox;
