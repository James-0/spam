$(document).ready(function () {
    const socket = io.connect('http://' + document.domain + ':' + location.port);
    var res = {}
    let allData = []

    // This function starts the train all algorithm job
    $('#train_all').click(function () {
        if ($('.results-section').is(":visible"))
            $(".results-section").hide();
        allData = []
        var $button = $(this);
        console.log('Train all algorithm button clicked');
        $('#results-table-body tbody').empty();
        $button.text('Training Data...');
        $button.prop('disabled', true);
        isStreamEnded = false;
        $(".results-section").show();
        socket.emit('generate_data');
    });





    socket.on('stream_data', function (data) {
        console.log('stream data has been called')
        // Check if the stream has ended
        if (!isStreamEnded) {
            updateTable(data);
            pushJsonIntoArrayMap(data)
            console.log(allData)
        }
        console.log('done called stream data')
    });

    socket.on('stream_end', function (data) {
        // Handle the end signal
        console.log(data.message);
        // Set the stream end flag
        isStreamEnded = true;
        $("#view-result").show();
        $('#train_all').prop('disabled', false).text('Train and Compare All Algorithms')
    });

    // Updating the table with the new data
    function updateTable(data) {
        algorithm = data.name || 'N/A';
        accuracy = data.accuracy || 'N/A';
        f1_score = data.f1_scoree || 'N/A';
        recall = data.recall || 'N/A';
        precision = data.precision || 'N/A';
        confusion_matrix = JSON.stringify(data.cf_matrix, null, 2) || 'N/A';
        // Simulate processing
        console.log("Processing algorithm:", data);
        // Add the row to the table
        var row = $("<tr></tr>");
        row.append(`<td>${algorithm}</td>`);
        row.append(`<td>${accuracy}</td>`);
        row.append(`<td>${f1_score}</td>`);
        row.append(`<td>${recall}</td>`);
        row.append(`<td>${precision}</td>`);
        row.append(`<div class="pull-left"><a href="#ex1" rel="modal:open"><button type=button class="btn btn-danger">View Metrics</button></a></div>`);
        row.append(`<div class="pull-right"><a href="#ex2" rel="modal:open"><button type=button class="btn btn-danger">Confusion Matrix</button></a></div>`);
        $("#results-table-body").append(row);
    }

    function pushJsonIntoArrayMap(jsonObject) {
        var map = new Map();
        Object.keys(jsonObject).forEach(function (key) {
            map.set(key, jsonObject[key]);
        });
        allData.push(map);
    }



    $('#myModal').on('show.bs.modal', function (event) {
        var modal = $(this);

        // var ctx = modal.find('#chartCanvas')[0].getContext('2d');
        // var canvas = document.getElementById('metrics-chart');
        // if (canvas.chart) {
        //     canvas.chart.destroy();
        // }


        // console.log('This is called')
        // const popupContent = $('<div>').addClass('popup-content')
        // const accuracy_data = allData.map(it => it.get('accuracy'))
        const accuracy_data = [0.8, 0.3, 0.93, 2.8]
        // console.log(allData)
        // console.log(accuracy_data)

        // const chartCanvas = $('<canvas>');
        // popupContent.append(chartCanvas);
        var ctx = document.getElementById('metrics-chart').getContext('2d');
        // const ctx = chartCanvas[0].getContext('2d');
        var chartctx = new Chart(ctx, {
            type: 'bar',
            data: {
                // labels: allData.map(obj => obj.get('name')),
                labels: ['Multi-NB', 'SVM', 'KNN', 'RF'],
                datasets: [{
                    label: "accuracy",
                    data: accuracy_data,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            },
        });

        // Display popup
        // $('.popup-content').remove(); // Clear previous content
        // $('body').append(popupContent);
    })
});