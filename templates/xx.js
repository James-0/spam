$(document).ready(function () {
    // Event handler for training buttons
    $(".train-btn").click(function () {
        // Store a reference to the button
        var $button = $(this);

        // Change button text
        $button.text('Training Data...');
        $button.prop('disabled', true);

        // Show progress bar container
        $("#progressContainer").show();

        // Simulate a process with a progress bar update
        var progressBar = $("#progressBar");
        var progressValue = 0;
        var interval = setInterval(function () {
            progressValue += 10;
            progressBar.width(progressValue + "%").attr("aria-valuenow", progressValue);

            // Check if the process is complete
            if (progressValue >= 100) {
                clearInterval(interval);
                // Hide progress bar container after completion
                $("#progressContainer").hide();

                // Fetch data from Flask using AJAX
                $.ajax({
                    type: 'GET',
                    url: '/get_results',  // Replace with the actual route in your Flask app
                    success: function (jsonData) {
                        // Assuming jsonData is an array of dictionaries with the structure described above

                        // Clear existing rows
                        $("#results-table-body").empty();

                        // Iterate through the algorithms in the JSON data
                        for (var i = 0; i < jsonData.length; i++) {
                            var algorithmData = jsonData[i];

                            // Create a new row
                            var row = $("<tr></tr>");

                            // Add cells for Algorithm, Accuracy, F1 Score, Image Data, etc.
                            row.append(`<td>${algorithmData.algorithm}</td>`);
                            row.append(`<td>${algorithmData.accuracy}</td>`);
                            row.append(`<td>${algorithmData.f1Score}</td>`);

                            // Append the row to the table
                            $("#results-table-body").append(row);
                        }

                        // Show the results section
                        $(".results-section").show();

                        // Enable the button
                        $button.prop('disabled', false);
                    },
                    error: function (error) {
                        console.log('Error fetching data from Flask:', error);
                        // Disable the button on error
                        $button.prop('disabled', true);
                    }
                });
            }
        }, 1000); // Adjust the interval based on your process duration
    });
});



function processAlgorithm(algorithmData) {
    var row = $("<tr></tr>");

    // Log algorithmData to the console for debugging
    console.log("algorithmData", typeof (algorithmData));

    // Add cells for Algorithm, Accuracy, F1 Score, Image Data, etc.
    row.append(`<td>${algorithmData["algorithm_name"]}</td>`);
    row.append(`<td>${algorithmData["acc_score"]}</td>`);
    row.append(`<td>${algorithmData["fo1_score"]}</td>`);

    // Append the row to the table
    $("#results-table-body").append(row);
}

function processAlgorithm(algorithmData) {
    var row = $("<tr></tr>");

    // Log algorithmData to the console for debugging
    console.log("algorithmData", typeof(algorithmData));

    // Add cells for Algorithm, Accuracy, F1 Score, Image Data, etc.
    row.append(`<td>${algorithmData["algorithm_name"]}</td>`);
    row.append(`<td>${algorithmData["acc_score"]}</td>`);
    row.append(`<td>${algorithmData["fo1_score"]}</td>`);

    // Append the row to the table
    $("#results-table-body").append(row);
}

