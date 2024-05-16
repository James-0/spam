$(document).ready(function () {
    const socket = io.connect('http://' + document.domain + ':' + location.port);
    let allData = []
    var processedData = null;
    $(".individual-button").hide();
    $("#view-result").hide();
    $(".chart-container").hide();
    $('#train_all').addClass('paused');

    // This function starts the train all algorithm job
    $('#train_all').click(function () {
        if ($('.results-section').is(":visible"))
            $(".results-section").hide();
        allData = []
        console.log(allData)
        var $button = $(this);
        $('#results-table-body tbody').empty();
        $button.text('Training Data...');
        $button.prop('disabled', true);
        $('#train_all').addClass('paused');
        isStreamEnded = false;
        $(".results-section").show();
        socket.emit('generate_data');
    });


    socket.on('stream_data', function (data) {
        if (!isStreamEnded) {
            pushJsonIntoArrayMap(data);
            updateTable(data);
        }
    });

    socket.on('stream_end', function (data) {
        console.log(data.message);
        isStreamEnded = true;
        $("#view-result").show();
        $('#train_all').removeClass('paused');
        $('#train_all').prop('disabled', false).text('Train and Compare All Algorithms')
    });

    socket.on('log_message', function (data) {
        console.log(data.message);
        const logContainer = $('#log-container');
        const logMessageElement = $('<p>').text(data.message).addClass('log-message');
        logContainer.append(logMessageElement);
        logContainer.scrollTop(logContainer.prop('scrollHeight'));
        setTimeout(function () {
            var dots = '';
            var dotCount = 1;
            var updateMessage = function () {
                logMessageElement.text(data.message + dots);
                dots += '.';
                dotCount++;
                if (dotCount <= 3) {
                    setTimeout(updateMessage, 1000);
                }
            };
            updateMessage();
        }, 5000);
    });

    function updateTable(data) {
        algorithm = data.name || 'N/A';
        accuracy = data.accuracy || 'N/A';
        f1_score = data.f1score || 'N/A';
        recall = data.recall || 'N/A';
        precision = data.precision || 'N/A';
        mae_train = data.mae_train || 'N/A';
        elapsed_time = data.elapsed_time || 'N/A';
        confusion_matrix = JSON.stringify(data.cf_matrix, null, 2) || 'N/A';
        // Add the row to the table
        var row = $("<tr></tr>");
        row.append(`<td>${algorithm}</td>`);
        row.append(`<td>${accuracy}</td>`);
        row.append(`<td>${f1_score}</td>`);
        row.append(`<td>${recall}</td>`);
        row.append(`<td>${precision}</td>`);
        row.append(`<td>${formatTime(elapsed_time)}</td>`);
        // row.append(`<td>${mae_train}</td>`);
        // row.append(`<div class="pull-left"><a href="#ex1" rel="modal:open"><button type=button class="btn btn-danger">View Metrics</button></a></div>`);
        // row.append(`<div class="pull-right"><a href="#ex2" rel="modal:open"><button type=button class="btn btn-danger">Confusion Matrix</button></a></div>`);
        $("#results-table-body").append(row);
    }

    function pushJsonIntoArrayMap(jsonObject) {
        var map = new Map();
        Object.keys(jsonObject).forEach(function (key) {
            map.set(key, jsonObject[key]);
        });
        allData.push(map);
    }

    // Function to create chart based on selected key
    function createChart(dataa, canvasId) {
        console.log(`div is ${canvasId + 'Chart'}`)
        var labels = ['Multi-NB', 'SVM', 'KNN', 'RF', 'AdaBoost'];
        var data = []

        if (canvasId == 'cf_matrix') {
            label = ['True Positive', 'False Positive', 'False Negative', 'True Negative'];
            dataa.forEach(function (matrix, index) {
                var counts = [];
                matrix.forEach(function (row) {
                    row.forEach(function (count) {
                        counts.push(count);
                    });
                });
                console.log(counts)
                data.push({
                    x: label,
                    y: counts,
                    type: 'bar',
                    name: labels[index],
                    text: counts.map(String),
                    textposition: 'auto',
                    hoverinfo: 'none',
                    marker: {
                    opacity: 0.9
                    }
                });
            });
            var layout = {
                barmode: 'group',
                title: 'Comparison of Confusion Matrices',
                xaxis: {
                    title: 'Classes'
                },
                yaxis: {
                    title: 'Count'
                },
                font: { size: 18 }
            };
            // Plotly.newPlot(canvasId + 'Chart', data, layout);
        } else {

            var trace1 = {
                x: labels,
                y: dataa,
                text: dataa.map(String),
                textposition: 'auto',
                hoverinfo: 'none',
                marker: {
                    color: ['rgba(31, 119, 180, 1)', 'rgba(255, 127, 14, 1)', 'rgba(44, 160, 44, 1)',
                        'rgba(214, 39, 40, 1)', 'rgba(148, 103, 189, 1)'],
                    opacity: 0.8,
                },
                type: 'bar',
            };

            var data = [trace1];

            var layout = {
                title: canvasId,
                font: { size: 18 }
            };

        }
        var config = {
            toImageButtonOptions: {
                format: 'png', // one of png, svg, jpeg, webp
                filename: 'Comparison of' + canvasId,
                height: 900,
                width: 600,
                scale: 1
            }
        };
        Plotly.newPlot(canvasId + 'Chart', data, layout, config);
    }

    $('#view-result').click(function () {
        console.log('Populating data in the tabs')
        processedData = processData()
        if (processedData) {
            Object.keys(processedData).forEach(function (key) {
                createChart(processedData[key], key);
            });
        }
        $(".chart-container").show();
    })


    function processData() {
        var keys = Array.from(allData[0].keys());
        keys = keys.filter(function (key) {
            return key !== 'name' && key !== 'model' && key !== 'mae_train';
        });
        var processedData = {};
        keys.forEach(function (key) {
            processedData[key] = allData.map(function (map) {
                return map.get(key);
            });
        });
        return processedData;
    }
    function formatTime(seconds) {
        if (seconds >= 60.0) {
            var minutes = Math.floor(seconds / 60);
            var remainingSeconds = Math.round(seconds % 60);
            return minutes + 'm ' + remainingSeconds + 's';
        } else {
            return seconds + 's'
        }
    }
});