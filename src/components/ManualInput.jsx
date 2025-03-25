import { useState } from "react";
import axios from "axios";
import { ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Chatbox from "./Chatbox";
import webIcon from "/webIcon.png";
import { Trash2Icon } from "lucide-react";
import { getRandomUserData } from "./dummyData";
import { ChevronDownIcon } from "lucide-react";
import { DatabaseIcon } from "lucide-react";

function ManualInput({
  backendEndpoint,
  parsedContent,
  setParsedContent,
  setJobListings,
  setSelectedCareerPath,
  setIsAnalyzing,
  setIsLoadingJobs,
  chatMessages,
  setChatMessages,
  formData,
  setFormData,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // New state to control whether the manual form is visible
  const [isFormExpanded, setIsFormExpanded] = useState(true);

  const handleAddField = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: [...prev[field], value],
    }));
  };

  const handleRemoveField = (field, index) => {
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].filter((_, i) => i !== index),
    }));
  };

  const handleChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleNestedChange = (field, index, key, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].map((item, i) =>
        i === index ? { ...item, [key]: value } : item
      ),
    }));
  };

  const handleAutoFill = () => {
    const randomData = getRandomUserData();
    setFormData({
      name: randomData.name,
      education: randomData.education,
      thesisAbstract: randomData.thesisAbstract,
      experience: randomData.experience,
      projects: randomData.projects,
      languages: randomData.additional.languages,
      softSkills: randomData.additional.softSkills,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const hasEducationData = formData.education.some(
      (edu) =>
        edu.school.trim() ||
        edu.degree.trim() ||
        edu.major.trim() ||
        edu.gpa.trim()
    );

    const hasExperienceData = formData.experience.some(
      (exp) => exp.title.trim() || exp.description.trim()
    );

    const hasProjectData = formData.projects.some(
      (proj) => proj.title.trim() || proj.description.trim()
    );

    if (!hasEducationData && !hasExperienceData && !hasProjectData) {
      setError(
        "Please provide at least one Education, Experience, or Project entry."
      );
      return;
    }
    setIsFormExpanded(false);
    setLoading(true);
    setError("");
    setTimeout(() => {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }, 100);

    setParsedContent(null);
    localStorage.removeItem("jobListings");
    setJobListings([]);
    setSelectedCareerPath("");

    setIsAnalyzing(true);
    setLoading(true);
    setIsLoadingJobs(false);
    setParsedContent(null);
    setChatMessages([
      {
        role: "assistant",
        content:
          "I can answer questions about your results. How can I help you?",
      },
    ]);

    const payload = {
      name: formData.name,
      education: formData.education,
      thesisAbstract: formData.thesisAbstract,
      experience: formData.experience,
      projects: formData.projects,
      additional: {
        languages: formData.languages,
        softSkills: formData.softSkills,
      },
    };

    try {
      const response = await axios.post(
        `${backendEndpoint}/api/parse-manual`,
        payload,
        {
          headers: {
            "Content-Type": "application/json",
            skip_zrok_interstitial: true,
          },
        }
      );
      setParsedContent(response.data);
      // Collapse the form upon successful analysis
      setIsFormExpanded(false);

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (err) {
      console.error("Error submitting manual input:", err);
      setError(
        "Error processing manual input. " + (err.response?.data || err.message)
      );
    } finally {
      setLoading(false);
      setIsAnalyzing(false);
    }
  };

  return (
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
          Fill in your resume details manually and get personalized career
          matches and professional insights.
        </p>
      </motion.div>

      {/* Chatbox Section */}
      {parsedContent && !loading && (
        <div className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 mt-4">
          <Chatbox
            parsedContent={parsedContent}
            messages={chatMessages}
            setMessages={setChatMessages}
          />
        </div>
      )}

      {/* Toggle Form Expand/Collapse */}
      {parsedContent && (
        <button
          onClick={() => setIsFormExpanded((prev) => !prev)}
          className="w-full bg-white border border-gray-100 text-gray-800 shadow-lg py-3 px-4 rounded-lg hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-300 font-medium text-sm tracking-wide transition-all flex items-center justify-start space-x-3"
        >
          <ChevronDownIcon
            size={18}
            className={`transition-transform duration-300 ${
              isFormExpanded ? "rotate-180" : ""
            }`}
          />
          <span className="text-left flex-grow">Modify Input Details</span>
        </button>
      )}

      {/* Form */}
      {isFormExpanded && (
        <>
          <AnimatePresence>
            <motion.div
              key="manualForm"
              initial={{ height: 0, opacity: 0, y: -20 }}
              animate={{ height: "auto", opacity: 1, y: 0 }}
              exit={{ height: 0, opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="bg-white shadow-xl rounded-lg p-8 border border-gray-100 overflow-hidden"
            >
              {/* Auto Fill Button */}
              <div className="flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="button"
                  onClick={handleAutoFill}
                  className="border-2 border-blue-500 text-blue-600 hover:bg-blue-50 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 flex items-center space-x-2"
                >
                  <DatabaseIcon size={16} />
                  <span>Auto Fill Test Data</span>
                </motion.button>
              </div>
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* Name */}
                <div>
                  <label className="block font-medium mb-2">Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => handleChange("name", e.target.value)}
                    className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                    required
                  />
                </div>

                {/* Education */}
                <div>
                  <label className="block font-medium mb-2">Education</label>
                  {formData.education.map((edu, idx) => (
                    <div
                      key={idx}
                      className="space-y-4 p-4 rounded-lg mb-4 bg-gray-50 border"
                    >
                      <input
                        type="text"
                        placeholder="School/University"
                        value={edu.school}
                        onChange={(e) =>
                          handleNestedChange(
                            "education",
                            idx,
                            "school",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      <select
                        value={edu.degree}
                        onChange={(e) =>
                          handleNestedChange(
                            "education",
                            idx,
                            "degree",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-3 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                        style={{ appearance: "none" }}
                      >
                        <option className="text-black/50" value="">
                          Select an Education Level
                        </option>
                        <option value="associate">High School</option>
                        <option value="associate">Associate</option>
                        <option value="bachelor's">Bachelor's</option>
                        <option value="master's">Master's</option>
                        <option value="doctoral">Doctoral</option>
                        <option value="other">Other</option>
                      </select>
                      <input
                        type="text"
                        placeholder="Major"
                        value={edu.major}
                        onChange={(e) =>
                          handleNestedChange(
                            "education",
                            idx,
                            "major",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      <input
                        type="text"
                        placeholder="GPA"
                        value={edu.gpa}
                        onChange={(e) =>
                          handleNestedChange(
                            "education",
                            idx,
                            "gpa",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      {formData.education.length > 1 && (
                        <button
                          type="button"
                          onClick={() => handleRemoveField("education", idx)}
                          className="text-red-500 hover:text-red-700 p-2 transition text-sm"
                        >
                          <Trash2Icon size={16} />
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      handleAddField("education", {
                        school: "",
                        degree: "",
                        major: "",
                        gpa: "",
                      })
                    }
                    className="text-blue-600 hover:text-blue-800 transition text-sm"
                  >
                    + Add Education
                  </button>
                </div>

                {/* Thesis Abstract */}
                <div>
                  <label className="block font-medium mb-2">
                    Thesis Abstract
                  </label>
                  <textarea
                    placeholder="Enter thesis abstract"
                    value={formData.thesisAbstract}
                    onChange={(e) =>
                      handleChange("thesisAbstract", e.target.value)
                    }
                    className="w-full border border-gray-300 h-56 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                  />
                </div>

                {/* Experience */}
                <div>
                  <label className="block font-medium mb-2">Experience</label>
                  {formData.experience.map((exp, idx) => (
                    <div
                      key={idx}
                      className="space-y-4 p-4 rounded-lg mb-4 bg-gray-50 border"
                    >
                      <input
                        type="text"
                        placeholder="Experience Title"
                        value={exp.title}
                        onChange={(e) =>
                          handleNestedChange(
                            "experience",
                            idx,
                            "title",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      <textarea
                        placeholder="Description"
                        value={exp.description}
                        onChange={(e) =>
                          handleNestedChange(
                            "experience",
                            idx,
                            "description",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 h-[200px] rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      {formData.experience.length > 1 && (
                        <button
                          type="button"
                          onClick={() => handleRemoveField("experience", idx)}
                          className="text-red-500 hover:text-red-700 p-2 transition text-sm"
                        >
                          <Trash2Icon size={16} />
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      handleAddField("experience", {
                        title: "",
                        description: "",
                      })
                    }
                    className="text-blue-600 hover:text-blue-800 transition text-sm"
                  >
                    + Add Experience
                  </button>
                </div>

                {/* Projects */}
                <div>
                  <label className="block font-medium mb-2">Projects</label>
                  {formData.projects.map((proj, idx) => (
                    <div
                      key={idx}
                      className="space-y-4 p-4 rounded-lg mb-4 bg-gray-50 border"
                    >
                      <input
                        type="text"
                        placeholder="Project Title"
                        value={proj.title}
                        onChange={(e) =>
                          handleNestedChange(
                            "projects",
                            idx,
                            "title",
                            e.target.value
                          )
                        }
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      <textarea
                        placeholder="Description"
                        value={proj.description}
                        onChange={(e) =>
                          handleNestedChange(
                            "projects",
                            idx,
                            "description",
                            e.target.value
                          )
                        }
                        className="w-full border h-[200px] border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                      />
                      {formData.projects.length > 1 && (
                        <button
                          type="button"
                          onClick={() => handleRemoveField("projects", idx)}
                          className="text-red-500 hover:text-red-700 p-2 transition text-sm"
                        >
                          <Trash2Icon size={16} />
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      handleAddField("projects", { title: "", description: "" })
                    }
                    className="text-blue-600 hover:text-blue-800 transition text-sm"
                  >
                    + Add Project
                  </button>
                </div>

                {/* Additional Information Section */}
                <div>
                  <div className="flex items-center my-6">
                    <div className="flex-grow border-t border-gray-300"></div>
                    <span className="mx-4 text-gray-500 text-sm uppercase tracking-wider">
                      Additional Information
                    </span>
                    <div className="flex-grow border-t border-gray-300"></div>
                  </div>

                  <div className="mb-6">
                    <label className="block text-sm font-semibold mb-2">
                      Languages
                    </label>
                    {formData.languages.map((lang, idx) => (
                      <div
                        key={idx}
                        className="flex items-center space-x-3 mb-3"
                      >
                        <input
                          type="text"
                          placeholder="Language"
                          value={lang}
                          onChange={(e) => {
                            const newLanguages = [...formData.languages];
                            newLanguages[idx] = e.target.value;
                            handleChange("languages", newLanguages);
                          }}
                          className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                        />
                        {formData.languages.length > 1 && (
                          <button
                            type="button"
                            onClick={() => handleRemoveField("languages", idx)}
                            className="text-red-500 hover:text-red-700 p-2 transition text-sm"
                          >
                            <Trash2Icon size={16} />
                          </button>
                        )}
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => handleAddField("languages", "")}
                      className="text-blue-600 hover:text-blue-800 transition text-sm"
                    >
                      + Add Language
                    </button>
                  </div>

                  <div>
                    <label className="block text-sm font-semibold mb-2">
                      Soft Skills
                    </label>
                    {formData.softSkills.map((skill, idx) => (
                      <div
                        key={idx}
                        className="flex items-center space-x-3 mb-3"
                      >
                        <input
                          type="text"
                          placeholder="Soft Skill"
                          value={skill}
                          onChange={(e) => {
                            const newSkills = [...formData.softSkills];
                            newSkills[idx] = e.target.value;
                            handleChange("softSkills", newSkills);
                          }}
                          className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                        />
                        {formData.softSkills.length > 1 && (
                          <button
                            type="button"
                            onClick={() => handleRemoveField("softSkills", idx)}
                            className="text-red-500 hover:text-red-700 p-2 transition text-sm"
                          >
                            <Trash2Icon size={16} />
                          </button>
                        )}
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => handleAddField("softSkills", "")}
                      className="text-blue-600 hover:text-blue-800 transition text-sm"
                    >
                      + Add Soft Skill
                    </button>
                  </div>
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-2 px-4 duration-150 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-400 flex items-center justify-center"
                >
                  {loading ? "Processing..." : "Analyze"}
                  <ArrowRight className="ml-2" size={18} />
                </button>
              </form>
              {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
            </motion.div>
          </AnimatePresence>
        </>
      )}

      {/* Analyze Button if Form is collapsed */}
      {!isFormExpanded && parsedContent && (
        <button
          onClick={handleSubmit}
          type="button"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 duration-150 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-400 flex items-center justify-center"
        >
          {loading ? "Processing..." : "Analyze"}
          <ArrowRight className="ml-2" size={18} />
        </button>
      )}

      {/* Loading Spinner */}
      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex justify-center items-center space-x-4 mt-6 text-blue-600"
          >
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            <p className="text-gray-600">Processing...</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ManualInput;
