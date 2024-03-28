var ctxP = document.getElementById("pieChart");
var myPieChart = new Chart(ctxP, {
       type: 'pie',
       data: {
              labels: ["Red", "Green", "Yellow", "Grey", "Dark Grey"],
              datasets: [{
                     data: [300, 50],
                     backgroundColor: ["#F7464A", "#46BFBD"],
                     hoverBackgroundColor: ["#FF5A5E", "#5AD3D1"]
              }]
       },
       options: {
              responsive: true
       }
});
