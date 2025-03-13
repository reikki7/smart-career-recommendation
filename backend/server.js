const express = require("express");
const multer = require("multer");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const cheerio = require("cheerio");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const app = express();
const port = process.env.PORT || 5000;
const lmStudioUrl = process.env.LM_STUDIO_URL || "http://localhost:1234";

app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: (cb) => {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + "-" + file.originalname);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"), false);
    }
  },
});

app.post("/api/parse-manual", async (req, res) => {
  try {
    const resumeData = req.body;
    const resumeText = JSON.stringify(resumeData, null, 2);

    const lmPayload = {
      model: "local-model",
      messages: [
        {
          role: "system",
          content: `Use the following data for RAG purposes:\n\n${resumeText}`,
        },
        {
          role: "user",
          content: `Please summarize who the person, and list some career path recommendations. The career path recommendations should be based on the document content and follow a VERY specific format which will be used for further processing in the frontend. If you're suggesting a career path, each field MUST NOT BE EMPTY. For Skills, it's the related skills to the career path based on the document and the recommended career path, so if you want to, for example, recommend Machine Learning Engineer, make sure I have the relevant experience from the document and also list only related skills in the JSON, like don't list PHP when the job recommendation is 3D Game Development, or Java for Machine Learning, like who uses Java for Machine Learning. And also in the JSON, for the skills, only list skillset and not the project, like if a career path is about Machine Learning, the skill would be like "Python" or "Tensorflow" and not "AI Chatbot", Relevant Projects/Experience will instead have its own array, it doesn't have to be more than 1 if the person only have 1 relevant project/experience, but make sure the ones listed are the exact same as the one in "Experience" array.\n\n
          
          Btw, Experience Title is the project/experience name, like what project the person was working on, or what the person did, like say it "Designer Portfolio Website" and NOT "Full-Stack Development", it should be specific and not what the project/experience is categorized in, also not "Teaching Assistant: Taught structural analysis techniques relevant to geotech engineering.", "Teaching Assistant" is enough, make it short. Make sure "relevantExperience" in jobRecommendation only lists the available experience from "Experience".\n\n       
IMPORTANT: Return only valid JSON without any additional text or commentary.
        
JSON Format:
{
  "summary": "[summary in a paragraph, not bullet points]",
  "Experience": [
    {
      "title": "[title of the experience1]",
      "description": "[description of the experience1]"
    },
    {
      "title": "[title of the experience2]",
      "description": "[description of the experience2]"
    },
    {
      "title": "[title of the experience3]",
      "description": "[description of the experience3]"
    },
    {
      "title": "[title of the experience4]",
      "description": "[description of the experience4]"
    },
    {
    (and so on... 4 is not the fixed number, just an example of formatting, if there are more, show more, if there are less, show less)
    }
  ],
  "jobRecommendation": [
    {
      "jobTitle": "[jobTitle1]" (String),
      "jobDescription": "[jobDescription1] (String)",
      "skills": "[skills1, skills1, skills1, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience1, relevantExperience1, relevantExperience1, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle2]" (String),
      "jobDescription": "[jobDescription2] (String)",
      "skills": "[skills2, skills2, skills2, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience2, relevantExperience2, relevantExperience2, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle3]" (String),
      "jobDescription": "[jobDescription3] (String)",
      "skills": "[skills3, skills3, skills3, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience3, relevantExperience3, relevantExperience3, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle4]" (String),
      "jobDescription": "[jobDescription4] (String)",
      "skills": "[skills4, skills4, skills4, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience4, relevantExperience4, relevantExperience4, ...] (Array of Strings)"
    },
      {
    (and so on... 4 is not the fixed number, just an example of formatting, if there are more, show more, if there are less, show less)
      }
  ]
}`,
        },
      ],
      temperature: 0.7,
    };

    const lmResponse = await axios.post(
      `${lmStudioUrl}/v1/chat/completions`,
      lmPayload
    );

    let lmFormatted = lmResponse.data.choices[0].message.content;
    lmFormatted = lmFormatted.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

    try {
      const startIndex = lmFormatted.indexOf("{");
      const endIndex = lmFormatted.lastIndexOf("}");
      if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
        const jsonPart = lmFormatted.substring(startIndex, endIndex + 1);
        const parsedJson = JSON.parse(jsonPart);
        res.setHeader("Content-Type", "application/json");
        res.send(JSON.stringify(parsedJson));
      } else {
        throw new Error("Could not extract valid JSON");
      }
    } catch (jsonError) {
      console.error("JSON extraction failed:", jsonError);
      res.status(500).send("Please try again, or check your input");
    }
  } catch (error) {
    console.error("Error processing manual resume data", error);
    res.status(500).send("Failed to process manual resume data");
  }
});

app.post("/api/parse-pdf", upload.single("pdfFile"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).send("No file uploaded");

    const pdfBuffer = fs.readFileSync(req.file.path);
    const data = await pdfParse(pdfBuffer);
    fs.unlinkSync(req.file.path);

    const lmPayload = {
      model: "local-model",
      messages: [
        {
          role: "system",
          content: `Use the following PDF document for RAG purposes:\n\n${data.text}`,
        },
        {
          role: "user",
          content: `Please summarize who the person, and list some career path recommendations. The career path recommendations should be based on the document content and follow a VERY specific format which will be used for further processing in the frontend. If you're suggesting a career path, each field MUST NOT BE EMPTY. For Skills, it's the related skills to the career path based on the document and the recommended career path, so if you want to, for example, recommend Machine Learning Engineer, make sure I have the relevant experience from the document and also list only related skills in the JSON, like don't list PHP when the job recommendation is 3D Game Development, or Java for Machine Learning, like who uses Java for Machine Learning. And also in the JSON, for the skills, only list skillset and not the project, like if a career path is about Machine Learning, the skill would be like "Python" or "Tensorflow" and not "AI Chatbot", Relevant Projects/Experience will instead have its own array, it doesn't have to be more than 1 if the person only have 1 relevant project/experience, but make sure the ones listed are the exact same as the one in "Experience" array.\n\n
          
          Btw, Experience Title is the project/experience name, like what project the person was working on, or what the person did, like say it "Designer Portfolio Website" and NOT "Full-Stack Development", it should be specific and not what the project/experience is categorized in, also not "Teaching Assistant: Taught structural analysis techniques relevant to geotech engineering.", "Teaching Assistant" is enough, make it short. Make sure "relevantExperience" in jobRecommendation only lists the available experience from "Experience".\n\n       
IMPORTANT: Return only valid JSON without any additional text or commentary.
        
JSON Format:
{
  "summary": "[summary in a paragraph, not bullet points]",
  "Experience": [
    {
      "title": "[title of the experience1]",
      "description": "[description of the experience1]"
    },
    {
      "title": "[title of the experience2]",
      "description": "[description of the experience2]"
    },
    {
      "title": "[title of the experience3]",
      "description": "[description of the experience3]"
    },
    {
      "title": "[title of the experience4]",
      "description": "[description of the experience4]"
    },
    {
    (and so on... 4 is not the fixed number, just an example of formatting, if there are more, show more, if there are less, show less)
    }
  ],
  "jobRecommendation": [
    {
      "jobTitle": "[jobTitle1]" (String),
      "jobDescription": "[jobDescription1] (String)",
      "skills": "[skills1, skills1, skills1, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience1, relevantExperience1, relevantExperience1, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle2]" (String),
      "jobDescription": "[jobDescription2] (String)",
      "skills": "[skills2, skills2, skills2, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience2, relevantExperience2, relevantExperience2, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle3]" (String),
      "jobDescription": "[jobDescription3] (String)",
      "skills": "[skills3, skills3, skills3, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience3, relevantExperience3, relevantExperience3, ...] (Array of Strings)"
    },
    {
      "jobTitle": "[jobTitle4]" (String),
      "jobDescription": "[jobDescription4] (String)",
      "skills": "[skills4, skills4, skills4, ...] (Array of Strings)",
      "relevantExperience": "[relevantExperience4, relevantExperience4, relevantExperience4, ...] (Array of Strings)"
    },
      {
    (and so on... 4 is not the fixed number, just an example of formatting, if there are more, show more, if there are less, show less)
      }
  ]
}`,
        },
      ],
      temperature: 0.7,
    };

    const lmResponse = await axios.post(
      `${lmStudioUrl}/v1/chat/completions`,
      lmPayload
    );

    let lmFormatted = lmResponse.data.choices[0].message.content;
    lmFormatted = lmFormatted.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

    try {
      const startIndex = lmFormatted.indexOf("{");
      const endIndex = lmFormatted.lastIndexOf("}");

      if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
        const jsonPart = lmFormatted.substring(startIndex, endIndex + 1);
        const parsedJson = JSON.parse(jsonPart);

        res.setHeader("Content-Type", "application/json");
        res.send(JSON.stringify(parsedJson));
      } else {
        throw new Error("Could not extract valid JSON");
      }
    } catch (jsonError) {
      console.error("JSON extraction failed:", jsonError);
      res.status(500).send("Please try again, or upload a different document");
    }
  } catch (error) {
    console.error("Error processing document", error);
    res.status(500).send("Failed to process document");
  }
});

app.get("/api/linkedin-jobs", async (req, res) => {
  try {
    const { title = "" } = req.query;
    const sanitizedTitle = title.replace(/[-:]/g, "");

    const linkedInUrl = `https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=${encodeURIComponent(
      sanitizedTitle
    )}&location=Indonesia&start=0`;

    // Fetch the HTML
    const response = await axios.get(linkedInUrl);
    const html = response.data;

    // Load into Cheerio to parse
    const $ = cheerio.load(html);
    const jobListings = [];

    $(".base-card").each((i, element) => {
      const jobTitle = $(element)
        .find(".base-search-card__title")
        .text()
        .trim();
      const organization = $(element)
        .find(".base-search-card__subtitle")
        .text()
        .trim();
      const location = $(element)
        .find(".job-search-card__location")
        .text()
        .trim();
      const url = $(element).find(".base-card__full-link").attr("href");
      const companyLogo = $(element)
        .find("img.artdeco-entity-image")
        .attr("data-delayed-url");
      const postedDate = $(element)
        .find("time.job-search-card__listdate")
        .attr("datetime");

      jobListings.push({
        title: jobTitle,
        organization,
        locations_derived: [location],
        url,
        companyLogo,
        date_posted: postedDate,
      });
    });

    res.json(jobListings);
  } catch (error) {
    console.error("Error fetching from LinkedIn:", error);
    res.status(500).json({ error: "Failed to fetch data from LinkedIn" });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { messages, parsedContent } = req.body;

    const systemMessage = {
      role: "system",
      content: `You have the following analysis results from the user's PDF:

${JSON.stringify(parsedContent, null, 2)}

Use these results to answer the user's queries in a concise way. Always refer back to the analysis context.`,
    };

    const conversation = [systemMessage, ...messages];

    const lmPayload = {
      model: "local-model",
      messages: conversation,
      temperature: 0.7,
    };

    const lmResponse = await axios.post(
      `${lmStudioUrl}/v1/chat/completions`,
      lmPayload
    );

    const assistantMessage = lmResponse.data.choices[0].message;
    assistantMessage.content = assistantMessage.content
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .trim();

    res.send(assistantMessage);
  } catch (error) {
    console.error("Error in /api/chat:", error);
    res.status(500).json({ error: "Failed to process chat request" });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
