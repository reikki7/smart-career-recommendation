import { motion, AnimatePresence } from "framer-motion";
import { Upload, ArrowRight, Briefcase } from "lucide-react";
import Chatbox from "./Chatbox";
import webIcon from "/webIcon.png";

const ResumeUpload = ({
  fileName,
  error,
  loading,
  isAnalyzing,
  isLoadingJobs,
  parsedContent,
  handleFileChange,
  handleUpload,
  chatMessages,
  setChatMessages,
}) => {
  return (
    <div
      className={`flex-1 flex flex-col duration-150 ${
        !parsedContent && "justify-center items-center"
      }`}
    >
      <div className="max-w-2xl mx-auto w-full px-4 py-12 space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-4xl font-bold text-blue-600 mb-4 flex justify-center items-center">
            <img src={webIcon} className="mr-3 w-10 h-10" alt="Jobify" />
            Jobify
          </h1>
          <p className="text-gray-600 max-w-md mx-auto">
            Upload your resume and get personalized career matches and
            professional insights
          </p>
        </motion.div>

        {parsedContent && !loading && (
          <div className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 mt-4">
            <Chatbox
              parsedContent={parsedContent}
              messages={chatMessages}
              setMessages={setChatMessages}
            />
          </div>
        )}

        {/* File Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="bg-white shadow-lg rounded-lg p-6 border border-gray-200">
            <div className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="dropzone-file"
                  className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer duration-150 bg-gray-50 hover:bg-gray-100"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 pt-2 mb-3 text-gray-500" />
                    <p className="mb-2 text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or
                      drag and drop
                    </p>
                    <p className="text-xs text-gray-500">PDF Document</p>
                    {fileName && (
                      <p className="mt-2 text-xs text-blue-600 font-semibold">
                        {fileName}
                      </p>
                    )}
                  </div>
                  <input
                    id="dropzone-file"
                    type="file"
                    className="hidden"
                    accept=".pdf"
                    onChange={handleFileChange}
                    disabled={loading || isAnalyzing}
                  />
                </label>
              </div>
              <button
                onClick={handleUpload}
                disabled={!fileName || loading || isAnalyzing}
                className="w-full bg-blue-600 duration-150 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-400 flex items-center justify-center"
              >
                {loading || isAnalyzing ? "Processing..." : "Analyze"}
                <ArrowRight className="ml-2" size={16} />
              </button>
            </div>
            {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
          </div>
        </motion.div>

        {/* Loading Animation */}
        <AnimatePresence mode="wait">
          {(loading || isLoadingJobs) && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex justify-center items-center space-x-3 mt-4 text-blue-600"
            >
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-current"></div>
              <p className="text-gray-600">
                {isAnalyzing
                  ? "Analyzing your document..."
                  : isLoadingJobs
                  ? "Seeking job applications..."
                  : "Processing..."}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default ResumeUpload;
