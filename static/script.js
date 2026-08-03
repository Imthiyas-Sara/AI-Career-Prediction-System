document.addEventListener("DOMContentLoaded", () => {
	const form = document.querySelector("form");
	const skills = document.getElementById("skills");
	const experience = document.getElementById("experience");
	const targetYear = document.getElementById("target_year");
	const education = document.getElementById("education");
	const submitBtn = document.querySelector(".predict-btn");

	const currentYear = new Date().getFullYear();

	// Helper: show error
	function setError(input, message) {
		input.classList.add("is-invalid");
		input.classList.remove("is-valid");

		let feedback = input.parentElement.querySelector(".invalid-feedback");

		if (!feedback) {
			feedback = document.createElement("div");
			feedback.className = "invalid-feedback";
			input.parentElement.appendChild(feedback);
		}

		feedback.textContent = message;
	}

	// Helper: clear error
	function clearError(input) {
		input.classList.remove("is-invalid");
		input.classList.add("is-valid");

		const feedback = input.parentElement.querySelector(".invalid-feedback");
		if (feedback) feedback.textContent = "";
	}

	// Validation rules
	function validateSkills() {
		const value = skills.value.trim();
		if (!value || value.length < 2) {
			setError(skills, "Please enter at least 2 characters for skills.");
			return false;
		}
		clearError(skills);
		return true;
	}

	function validateExperience() {
		const value = experience.value.trim();

		if (value === "") {
			setError(experience, "Experience is required.");
			return false;
		}

		const num = Number(value);

		if (isNaN(num)) {
			setError(experience, "Experience must be a number.");
			return false;
		}

		if (num < 0) {
			setError(experience, "Experience cannot be negative.");
			return false;
		}

		clearError(experience);
		return true;
	}

	function validateYear() {
		const value = targetYear.value.trim();

		if (value === "") {
			setError(targetYear, "Target year is required.");
			return false;
		}

		const num = Number(value);

		if (isNaN(num)) {
			setError(targetYear, "Year must be a number.");
			return false;
		}

		if (num < currentYear) {
			setError(targetYear, `Year must be ${currentYear} or later.`);
			return false;
		}

		clearError(targetYear);
		return true;
	}

	function validateEducation() {
		const value = education.value;

		if (!value || value.includes("Select")) {
			setError(education, "Please select your education level.");
			return false;
		}

		clearError(education);
		return true;
	}

	// Master validation
	function validateForm() {
		const a = validateSkills();
		const b = validateExperience();
		const c = validateYear();
		const d = validateEducation();

		const isValid = a && b && c && d;

		submitBtn.disabled = !isValid;
		return isValid;
	}

	// Real-time validation
	skills.addEventListener("input", validateForm);
	experience.addEventListener("input", validateForm);
	targetYear.addEventListener("input", validateForm);
	education.addEventListener("change", validateForm);

	// Submit handler
	form.addEventListener("submit", (e) => {
		if (!validateForm()) {
			e.preventDefault();
		}
	});

	// Initial state
	validateForm();
});