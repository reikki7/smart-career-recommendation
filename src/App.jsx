import { useState, useEffect, useRef } from "react";
import axios from "axios";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import { FileText, Briefcase } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

import ManualInput from "./components/ManualInput";
import ResumeUpload from "./components/ResumeUpload";
import JobListings from "./components/JobListings";
import AnalysisResult from "./components/AnalysisResult";

dayjs.extend(relativeTime);

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [pdfContent, setPdfContent] = useState(null);
  const [parsedContent, setParsedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobListings, setJobListings] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [vennDiagramRendered, setVennDiagramRendered] = useState(false);
  const [selectedCareerPath, setSelectedCareerPath] = useState("");
  const [inputMode, setInputMode] = useState("manual");
  const [isSticky, setIsSticky] = useState(false);
  const [showLeftSidebar, setShowLeftSidebar] = useState(false);
  const [showRightSidebar, setShowRightSidebar] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      content: "I can answer questions about your results. How can I help you?",
    },
  ]);
  const [manualFormData, setManualFormData] = useState({
    name: "",
    education: [{ school: "", degree: "", major: "", gpa: "" }],
    thesisAbstract: "",
    experience: [{ title: "", description: "" }],
    projects: [{ title: "", description: "" }],
    languages: [""],
    softSkills: [""],
  });

  const tabHeaderRef = useRef(null);

  const backendEndpoint =
    import.meta.env.VITE_BACKEND_API_URL || "http://localhost:5000";

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };

    checkMobile();

    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError("");
    } else {
      setFile(null);
      setFileName("");
      setError("Please select a PDF file");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a PDF file first");
      return;
    }

    localStorage.removeItem("jobListings");
    setJobListings([]);
    setSelectedCareerPath("");

    setIsAnalyzing(true);
    setLoading(true);
    setIsLoadingJobs(false);
    setPdfContent(null);
    setParsedContent(null);

    const formData = new FormData();
    formData.append("pdfFile", file);

    try {
      const response = await axios.post(
        `${backendEndpoint}/api/parse-pdf`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            skip_zrok_interstitial: true,
          },
        }
      );

      if (typeof response.data === "object") {
        console.log("Received valid JSON response:", response.data);
        setParsedContent(response.data);
        setIsLoadingJobs(true);
      } else {
        try {
          const jsonData = JSON.parse(response.data);
          setParsedContent(jsonData);
          setIsLoadingJobs(true);
        } catch (jsonError) {
          console.error("Error parsing response:", jsonError);
          const cleanedOutput = response.data.replace(/```/g, "");
          console.log("Cleaned non-JSON output:", cleanedOutput);
          setPdfContent(cleanedOutput);
        }
      }

      setError("");
    } catch (err) {
      console.error("Error during PDF upload:", err);
      const errorMessage = err.response?.data || err.message;
      setError("Error parsing document: " + errorMessage);
      setPdfContent(null);
      setParsedContent(null);
      setIsLoadingJobs(false);
    } finally {
      setIsAnalyzing(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (parsedContent) {
      setVennDiagramRendered(false);
      setIsLoadingJobs(true);

      if (isMobile) {
        setShowLeftSidebar(true);
      }
    } else if (pdfContent) {
      try {
        const cleanedJson = pdfContent
          .trim()
          .replace(/```/g, "")
          .replace(/^json\s*:?/i, "");
        const jsonEndIndex = cleanedJson.lastIndexOf("}");
        if (jsonEndIndex !== -1) {
          const jsonPart = cleanedJson.substring(0, jsonEndIndex + 1);
          const jsonData = JSON.parse(jsonPart);
          setParsedContent(jsonData);
          setVennDiagramRendered(false);
          setIsLoadingJobs(true);
        } else {
          throw new Error("Could not find valid JSON structure");
        }
      } catch (err) {
        console.error("Error parsing JSON:", err);
        setError(
          "We couldn't parse the data from your PDF. Please try again or upload a different PDF."
        );
        setParsedContent(null);
        setIsLoadingJobs(false);
      }
    }
  }, [pdfContent, parsedContent]);

  useEffect(() => {
    // Calculate header height once on mount
    const headerHeight = tabHeaderRef.current?.offsetHeight || 0;

    const handleScroll = () => {
      if (!tabHeaderRef.current) return;
      setIsSticky(window.scrollY > tabHeaderRef.current.offsetTop);
    };

    // Add placeholder with exact header height
    if (isSticky) {
      const placeholder = document.createElement("div");
      placeholder.style.height = `${headerHeight}px`;
      placeholder.id = "header-placeholder";
      tabHeaderRef.current.parentNode.insertBefore(
        placeholder,
        tabHeaderRef.current
      );
    } else {
      const placeholder = document.getElementById("header-placeholder");
      if (placeholder) placeholder.remove();
    }

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [isSticky]);

  useEffect(() => {
    async function fetchJobListings(title) {
      const sanitizedTitle = title.replace(/[-:]/g, "");
      try {
        const response = await axios.get(
          `${backendEndpoint}/api/linkedin-jobs`,
          {
            headers: {
              "Content-Type": "application/json",
              "ngrok-skip-browser-warning": true,
              skip_zrok_interstitial: true,
            },
            params: { title: sanitizedTitle },
          }
        );
        console.log("Fetched job listings for", title, response.data);

        const jobsWithCareerPath = response.data.map((job) => ({
          ...job,
          careerPath: title,
        }));

        return jobsWithCareerPath;
      } catch (error) {
        console.error("Error fetching job listings for", title, error);
        return [];
      }
    }

    if (parsedContent?.jobRecommendation?.length && isLoadingJobs) {
      const cachedListings = localStorage.getItem("jobListings");
      if (cachedListings) {
        setJobListings(JSON.parse(cachedListings));
        setIsLoadingJobs(false);
        setVennDiagramRendered(true);
      } else {
        const uniqueTitles = [
          ...new Set(
            parsedContent.jobRecommendation.map((job) => job.jobTitle)
          ),
        ];

        Promise.all(uniqueTitles.map(fetchJobListings)).then((results) => {
          const flattened = results.flat();
          setJobListings(flattened);
          localStorage.setItem("jobListings", JSON.stringify(flattened));
          setIsLoadingJobs(false);
          setVennDiagramRendered(true);
        });
      }
    }
  }, [parsedContent, backendEndpoint, isLoadingJobs]);

  const handleCareerPathClick = (careerPath) => {
    if (selectedCareerPath === careerPath) {
      setSelectedCareerPath("");
    } else {
      setSelectedCareerPath(careerPath);
    }
  };

  const toggleLeftSidebar = () => {
    setShowLeftSidebar(!showLeftSidebar);
    if (!showLeftSidebar) setShowRightSidebar(false);
  };

  const toggleRightSidebar = () => {
    setShowRightSidebar(!showRightSidebar);
    if (!showRightSidebar) setShowLeftSidebar(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex relative">
      {/* Mobile Toggle Buttons */}
      {isMobile && parsedContent && (
        <div className="fixed bottom-4 left-0 right-0 flex justify-center z-30 gap-4">
          <button
            onClick={toggleLeftSidebar}
            className={`rounded-full p-3 shadow-lg ${
              showLeftSidebar
                ? "bg-blue-600 text-white"
                : "bg-white text-blue-600"
            }`}
            aria-label="Toggle Analysis Results"
          >
            <FileText size={24} />
          </button>
          <button
            onClick={() => {
              setShowLeftSidebar(false);
              setShowRightSidebar(false);
            }}
            className={`rounded-full p-3 shadow-lg  ${
              showLeftSidebar || showRightSidebar
                ? "bg-white text-blue-600"
                : "bg-blue-600 text-white"
            }`}
            aria-label="Home"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
          <button
            onClick={toggleRightSidebar}
            className={`rounded-full p-3 shadow-lg ${
              showRightSidebar
                ? "bg-blue-600 text-white"
                : "bg-white text-blue-600"
            }`}
            aria-label="Toggle Job Listings"
          >
            <Briefcase size={24} />
          </button>
        </div>
      )}

      {/* Left Sidebar (Analysis Results) */}
      <AnimatePresence mode="wait">
        {((isMobile && showLeftSidebar) || (!isMobile && parsedContent)) && (
          <motion.div
            key="leftSidebar"
            initial={isMobile ? { opacity: 0, x: -100 } : { opacity: 0 }}
            animate={isMobile ? { opacity: 1, x: 0 } : { opacity: 1 }}
            exit={isMobile ? { opacity: 0, x: -100 } : { opacity: 0 }}
            transition={{ duration: 0.3 }}
            className={`${isMobile ? "absolute inset-0 z-20" : "w-1/4"} ${
              !isMobile ? "border-r border-gray-200" : ""
            }`}
            style={isMobile ? { width: "100%" } : {}}
          >
            <AnalysisResult
              parsedContent={parsedContent}
              onCareerPathClick={handleCareerPathClick}
              selectedCareerPath={selectedCareerPath}
              isMobile={isMobile}
              onClose={toggleLeftSidebar}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Section */}
      <div
        className={`flex-1 flex flex-col ${
          isMobile && (showLeftSidebar || showRightSidebar) ? "hidden" : ""
        }`}
      >
        {isSticky && <div className="h-12"></div>}
        {/* Tab Header */}
        <div
          ref={tabHeaderRef}
          className={`flex justify-center space-x-4 border-b border-gray-200 bg-gray-50 z-10 w-full ${
            isSticky ? "fixed top-0 left-0 right-0" : ""
          }`}
        >
          <button
            onClick={() => {
              setInputMode("manual");
            }}
            className={`py-2 px-4 font-semibold border-b-2 ${
              inputMode === "manual"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-600"
            }`}
          >
            Manual Input
          </button>
          <button
            onClick={() => {
              setInputMode("upload");
            }}
            className={`py-2 px-4 font-semibold border-b-2 ${
              inputMode === "upload"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-600"
            }`}
          >
            Resume Upload
          </button>
        </div>

        {inputMode === "upload" ? (
          <ResumeUpload
            fileName={fileName}
            error={error}
            loading={loading}
            isAnalyzing={isAnalyzing}
            isLoadingJobs={isLoadingJobs}
            parsedContent={parsedContent}
            vennDiagramRendered={vennDiagramRendered}
            setVennDiagramRendered={setVennDiagramRendered}
            handleFileChange={handleFileChange}
            handleUpload={handleUpload}
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
          />
        ) : (
          <ManualInput
            backendEndpoint={backendEndpoint}
            setParsedContent={setParsedContent}
            parsedContent={parsedContent}
            setIsLoadingJobs={setIsLoadingJobs}
            setJobListings={setJobListings}
            setSelectedCareerPath={setSelectedCareerPath}
            setIsAnalyzing={setIsAnalyzing}
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
            formData={manualFormData}
            setFormData={setManualFormData}
          />
        )}
      </div>

      {/* Right Sidebar (Job Listings) */}
      <AnimatePresence mode="wait">
        {((isMobile && showRightSidebar) ||
          (!isMobile && jobListings.length > 0)) && (
          <motion.div
            key="rightSidebar"
            initial={isMobile ? { opacity: 0, x: 100 } : { opacity: 0 }}
            animate={isMobile ? { opacity: 1, x: 0 } : { opacity: 1 }}
            exit={isMobile ? { opacity: 0, x: 100 } : { opacity: 0 }}
            transition={{ duration: 0.3 }}
            className={`${isMobile ? "absolute inset-0 z-20" : "w-1/4"} ${
              !isMobile ? "border-l border-gray-200" : ""
            }`}
            style={isMobile ? { width: "100%" } : {}}
          >
            <JobListings
              jobListings={jobListings}
              selectedCareerPath={selectedCareerPath}
              setSelectedCareerPath={setSelectedCareerPath}
              isMobile={isMobile}
              onClose={toggleRightSidebar}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
