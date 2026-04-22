const express = require("express");
const multer = require("multer");
const { handlePrediction } = require("../controllers/predictController");

const router = express.Router();

const upload = multer({ dest: "uploads/" });

router.post("/", upload.single("image"), (req, res, next) => {
  console.log("🔥 Route hit");
  next();
}, handlePrediction);

router.post("/ask", async (req, res) => {
  try {
    const response = await axios.post(
      "http://127.0.0.1:8001/ask",
      req.body
    );

    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: "Q&A failed" });
  }
});
module.exports = router;