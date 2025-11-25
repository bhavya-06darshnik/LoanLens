async function predictLoan() {

    const data = {
        current_loan_amount: parseFloat(document.getElementById("current_loan_amount").value),
        term: document.getElementById("term").value,
        credit_score: parseFloat(document.getElementById("credit_score").value),
        years_in_current_job: parseFloat(document.getElementById("years_in_current_job").value),
        home_ownership: document.getElementById("home_ownership").value,
        annual_income: parseFloat(document.getElementById("annual_income").value),
        purpose: document.getElementById("purpose").value,
        monthly_debt: parseFloat(document.getElementById("monthly_debt").value),
        years_of_credit_history: parseFloat(document.getElementById("years_of_credit_history").value),
        months_since_last_delinquent: parseFloat(document.getElementById("months_since_last_delinquent").value),
        number_of_open_accounts: parseFloat(document.getElementById("number_of_open_accounts").value),
        number_of_credit_problems: parseFloat(document.getElementById("number_of_credit_problems").value),
        current_credit_balance: parseFloat(document.getElementById("current_credit_balance").value),
        maximum_open_credit: parseFloat(document.getElementById("maximum_open_credit").value),
        bankruptcies: parseFloat(document.getElementById("bankruptcies").value),
        tax_liens: parseFloat(document.getElementById("tax_liens").value)
    };

    const payload = {
        data: data,
        threshold: { threshold_metrics: "precision" }
    };

    const response = await fetch("http://localhost:8000/predict_explaination", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const result = await response.json();

    document.getElementById("resultBox").innerHTML =
        `<b>${result.prediction.prediction[0]}</b>`;

    const influence = result.feature_influence;

    // Draw SHAP influence chart
    const ctx = document.getElementById("shapChart").getContext("2d");

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(influence),
            datasets: [{
                label: "Feature Influence",
                data: Object.values(influence),
                backgroundColor: "rgba(62, 107, 236, 0.6)"
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { ticks: { maxRotation: 90, minRotation: 60 } }
            }
        }
    });
}
