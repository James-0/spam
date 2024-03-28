import "../static/bootstrap/css/bootstrap.css";
import "../static/bootstrap/css/bootstrap-grid.min.css";


const handleClick = (event: MouseEvent) => {
    console.log(event);
    alert('Hello, it is here');
  };
  
  document.getElementById("button")?.addEventListener("click", handleClick);