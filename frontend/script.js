// ================================================================
// INTENTSTORE FRONTEND
// script.js
// ================================================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("INTENTSTORE frontend loaded");


    // ============================================================
    // ELEMENTS
    // ============================================================

    const questionInput =
        document.querySelector("input, textarea");

    const searchButton =
        findElement([
            "#searchBtn",
            "#search-button",
            "#searchButton",
            "button"
        ]);

    const modeSelect =
        document.querySelector("select");

    const exampleButton =
        findElement([
            "#exampleBtn",
            "#example-button",
            "#exampleButton"
        ]);


    // ============================================================
    // FIND RESULT CONTAINER
    // ============================================================

    let resultContainer =
        document.querySelector(
            "#results"
        );

    if (!resultContainer) {

        resultContainer =
            document.querySelector(
                ".results"
            );
    }

    if (!resultContainer) {

        resultContainer =
            document.createElement(
                "div"
            );

        resultContainer.id =
            "results";

        resultContainer.style.marginTop =
            "30px";

        document.body.appendChild(
            resultContainer
        );
    }


    // ============================================================
    // CRITICAL:
    // DO NOT SEARCH WHILE TYPING
    // ============================================================

    if (questionInput) {

        questionInput.addEventListener(
            "input",
            () => {

                // Intentionally empty.

                // DO NOT CALL API HERE.
                //
                // Information retrieval happens
                // ONLY when Search is clicked.

            }
        );

        questionInput.addEventListener(
            "paste",
            () => {

                // Do nothing.
                //
                // Pasting a question must NOT
                // automatically trigger search.

            }
        );
    }


    // ============================================================
    // SEARCH BUTTON
    // ============================================================

    if (searchButton) {

        searchButton.addEventListener(
            "click",
            async (event) => {

                event.preventDefault();

                await performSearch();

            }
        );

    } else {

        console.error(
            "Search button was not found."
        );
    }


    // ============================================================
    // ENTER KEY
    // ============================================================

    if (questionInput) {

        questionInput.addEventListener(
            "keydown",
            async (event) => {

                if (
                    event.key === "Enter"
                    &&
                    !event.shiftKey
                ) {

                    event.preventDefault();

                    await performSearch();

                }

            }
        );

    }


    // ============================================================
    // EXAMPLE QUESTION
    // ============================================================

    if (exampleButton) {

        exampleButton.addEventListener(
            "click",
            async (event) => {

                event.preventDefault();

                await loadExampleQuestion();

            }
        );

    }


    // ============================================================
    // SEARCH
    // ============================================================

    async function performSearch() {

        if (!questionInput) {

            showError(
                "Question input was not found."
            );

            return;
        }

        const question =
            questionInput.value.trim();

        if (!question) {

            showError(
                "Please enter a question."
            );

            return;
        }


        const mode =
            modeSelect
                ? modeSelect.value
                : "";


        console.log(
            "SEARCH CLICKED"
        );

        console.log(
            "Mode:",
            mode
        );

        console.log(
            "Question:",
            question
        );


        // --------------------------------------------------------
        // DOCUMENT RETRIEVAL
        // --------------------------------------------------------

        if (
            isDocumentMode(mode)
        ) {

            await retrieveDocument(
                question
            );

            return;
        }


        // --------------------------------------------------------
        // INFORMATION RETRIEVAL
        // --------------------------------------------------------

        await retrieveInformation(
            question
        );

    }


    // ============================================================
    // INFORMATION RETRIEVAL
    // ============================================================

    async function retrieveInformation(
        question
    ) {

        setLoading(
            "Searching the knowledge base..."
        );

        try {

            const response =
                await fetch(
                    "/api/ask",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question:
                                question
                        })

                    }
                );


            const data =
                await readJSON(
                    response
                );


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    data.answer
                    ||
                    "Information retrieval failed."
                );

            }


            displayInformation(
                data
            );


        } catch (error) {

            console.error(
                "INFORMATION RETRIEVAL ERROR:",
                error
            );

            showError(
                error.message
            );

        }

    }


    // ============================================================
    // DOCUMENT RETRIEVAL
    // ============================================================

    async function retrieveDocument(
        query
    ) {

        setLoading(
            "Finding document..."
        );

        try {

            console.log(
                "Calling /api/document"
            );

            const response =
                await fetch(
                    "/api/document",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            query:
                                query
                        })

                    }
                );


            const data =
                await readJSON(
                    response
                );


            console.log(
                "Document response:",
                data
            );


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    "Document was not found."
                );

            }


            displayDocument(
                data.document
            );


        } catch (error) {

            console.error(
                "DOCUMENT RETRIEVAL ERROR:",
                error
            );

            showError(
                error.message
            );

        }

    }


    // ============================================================
    // SAFE JSON READER
    // ============================================================

    async function readJSON(
        response
    ) {

        const text =
            await response.text();

        console.log(
            "SERVER RESPONSE:",
            text
        );


        try {

            return JSON.parse(
                text
            );

        } catch (error) {

            console.error(
                "SERVER DID NOT RETURN JSON:",
                text
            );

            throw new Error(
                "The backend returned HTML instead of JSON. "
                +
                "Check that Flask is running on "
                +
                "127.0.0.1:5000."
            );

        }

    }


    // ============================================================
    // DISPLAY INFORMATION
    // ============================================================

    function displayInformation(
        data
    ) {

        resultContainer.innerHTML = "";


        const answer =
            document.createElement(
                "div"
            );

        answer.className =
            "retrieval-answer";


        answer.innerHTML =
            `
            <h2>Answer</h2>
            <p>${escapeHTML(
                data.answer || ""
            )}</p>
            `;


        resultContainer.appendChild(
            answer
        );


        const results =
            data.results || [];


        if (!results.length) {

            const empty =
                document.createElement(
                    "div"
                );

            empty.innerHTML =
                `
                <p>No supporting evidence found.</p>
                `;

            resultContainer.appendChild(
                empty
            );

            return;
        }


        const heading =
            document.createElement(
                "h2"
            );

        heading.textContent =
            "Retrieved Evidence";

        resultContainer.appendChild(
            heading
        );


        results.forEach(
            (result, index) => {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "evidence-card";


                card.innerHTML =
                    `
                    <h3>
                        Source ${index + 1}
                    </h3>

                    <p>
                        <strong>Document:</strong>
                        ${escapeHTML(
                            result.filename
                            || "Unknown"
                        )}
                    </p>

                    <p>
                        <strong>Page:</strong>
                        ${escapeHTML(
                            String(
                                result.page_number
                                || "Unknown"
                            )
                        )}
                    </p>

                    <p>
                        ${escapeHTML(
                            result.text
                            || ""
                        )}
                    </p>
                    `;


                resultContainer.appendChild(
                    card
                );

            }
        );

    }


    // ============================================================
    // DISPLAY DOCUMENT
    // ============================================================

    function displayDocument(
        documentData
    ) {

        resultContainer.innerHTML = "";


        const card =
            document.createElement(
                "div"
            );

        card.className =
            "document-result";


        const filename =
            documentData.filename
            ||
            "Document";


        const url =
            documentData.url
            ||
            "";


        card.innerHTML =
            `
            <h2>Document Found</h2>

            <p>
                <strong>Filename:</strong>
                ${escapeHTML(
                    filename
                )}
            </p>

            <p>
                <strong>Type:</strong>
                ${escapeHTML(
                    documentData.extension
                    || ""
                )}
            </p>

            <div style="
                margin-top:20px;
                display:flex;
                gap:15px;
                flex-wrap:wrap;
            ">

                <a
                    href="${escapeAttribute(url)}"
                    target="_blank"
                    rel="noopener"
                    style="
                        display:inline-block;
                        padding:14px 24px;
                        border-radius:12px;
                        background:#171717;
                        color:white;
                        text-decoration:none;
                    "
                >
                    Open Document
                </a>

                <a
                    href="${escapeAttribute(url)}"
                    download
                    style="
                        display:inline-block;
                        padding:14px 24px;
                        border-radius:12px;
                        border:1px solid #999;
                        color:#171717;
                        text-decoration:none;
                    "
                >
                    Download
                </a>

            </div>
            `;


        resultContainer.appendChild(
            card
        );

    }


    // ============================================================
    // EXAMPLE QUESTION
    // ============================================================

    async function loadExampleQuestion() {

        try {

            const response =
                await fetch(
                    "/api/examples"
                );


            const data =
                await readJSON(
                    response
                );


            if (
                !data.success
                ||
                !data.questions
                ||
                !data.questions.length
            ) {

                throw new Error(
                    "No example questions available."
                );

            }


            const questions =
                data.questions;


            const randomQuestion =
                questions[
                    Math.floor(
                        Math.random()
                        *
                        questions.length
                    )
                ];


            questionInput.value =
                randomQuestion;


            // IMPORTANT:
            // Do NOT search automatically.

            resultContainer.innerHTML =
                "";

        } catch (error) {

            console.error(
                error
            );

            showError(
                error.message
            );

        }

    }


    // ============================================================
    // DOCUMENT MODE DETECTION
    // ============================================================

    function isDocumentMode(
        mode
    ) {

        const value =
            String(
                mode || ""
            ).toLowerCase();


        return (
            value.includes(
                "document"
            )
        );

    }


    // ============================================================
    // LOADING
    // ============================================================

    function setLoading(
        message
    ) {

        resultContainer.innerHTML =
            `
            <div class="loading">
                ${escapeHTML(
                    message
                )}
            </div>
            `;

    }


    // ============================================================
    // ERROR
    // ============================================================

    function showError(
        message
    ) {

        resultContainer.innerHTML =
            `
            <div
                style="
                    padding:20px;
                    border:1px solid #e58b7c;
                    border-radius:16px;
                    background:#f9d8d2;
                    color:#8b2418;
                    margin-top:20px;
                "
            >
                ${escapeHTML(
                    message
                )}
            </div>
            `;

    }


    // ============================================================
    // ELEMENT FINDER
    // ============================================================

    function findElement(
        selectors
    ) {

        for (
            const selector of selectors
        ) {

            const element =
                document.querySelector(
                    selector
                );

            if (element) {

                return element;

            }

        }

        return null;

    }


    // ============================================================
    // HTML ESCAPING
    // ============================================================

    function escapeHTML(
        value
    ) {

        return String(
            value ?? ""
        )
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

    }


    function escapeAttribute(
        value
    ) {

        return escapeHTML(
            value
        );

    }

});